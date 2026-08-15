use std::io::{self, Write};
use std::process::Stdio;
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use rmcp::RoleClient;
use rmcp::service::{RxJsonRpcMessage, TxJsonRpcMessage};
use rmcp::transport::Transport;
use rmcp::transport::async_rw::JsonRpcMessageCodec;
use serde::Serialize;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio_util::codec::{FramedRead, FramedWrite};

type ClientReader<R> = FramedRead<R, JsonRpcMessageCodec<RxJsonRpcMessage<RoleClient>>>;
type ClientWriter<W> = FramedWrite<W, JsonRpcMessageCodec<TxJsonRpcMessage<RoleClient>>>;

struct BoundedIoTransport<R, W> {
    reader: ClientReader<R>,
    writer: Arc<Mutex<Option<ClientWriter<W>>>>,
    max_message_bytes: usize,
}

impl<R, W> BoundedIoTransport<R, W>
where
    R: AsyncRead + Send + Unpin,
    W: AsyncWrite + Send + Unpin + 'static,
{
    fn new(reader: R, writer: W, max_message_bytes: usize) -> Self {
        Self {
            reader: FramedRead::new(
                reader,
                JsonRpcMessageCodec::new_with_max_length(max_message_bytes),
            ),
            writer: Arc::new(Mutex::new(Some(FramedWrite::new(
                writer,
                JsonRpcMessageCodec::new(),
            )))),
            max_message_bytes,
        }
    }
}

impl<R, W> Transport<RoleClient> for BoundedIoTransport<R, W>
where
    R: AsyncRead + Send + Unpin,
    W: AsyncWrite + Send + Unpin + 'static,
{
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let writer = self.writer.clone();
        let max_message_bytes = self.max_message_bytes;
        async move {
            validate_outbound_message(&item, max_message_bytes)?;
            let mut writer = writer.lock().await;
            let writer = writer.as_mut().ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotConnected, "MCP stdio transport closed")
            })?;
            writer.send(item).await.map_err(Into::into)
        }
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleClient>> {
        match self.reader.next().await {
            Some(Ok(message)) => Some(message),
            Some(Err(_)) => None,
            None => None,
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.writer.lock().await.take();
        Ok(())
    }
}

fn validate_outbound_message<T: Serialize>(item: &T, maximum: usize) -> io::Result<()> {
    let mut writer = SizeBoundWriter::new(maximum);
    match serde_json::to_writer(&mut writer, item) {
        Ok(()) => Ok(()),
        Err(_) if writer.exceeded => Err(message_too_large(maximum)),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "MCP stdio message could not be serialized",
        )),
    }
}

fn message_too_large(maximum: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("MCP stdio message exceeded {maximum} bytes"),
    )
}

struct SizeBoundWriter {
    remaining: usize,
    exceeded: bool,
}

impl SizeBoundWriter {
    fn new(maximum: usize) -> Self {
        Self {
            remaining: maximum,
            exceeded: false,
        }
    }
}

impl Write for SizeBoundWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.len() > self.remaining {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MCP stdio message exceeded the configured limit",
            ));
        }
        self.remaining -= buffer.len();
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) struct BoundedChildProcess {
    child: Option<Child>,
    transport: BoundedIoTransport<ChildStdout, ChildStdin>,
}

impl BoundedChildProcess {
    pub(crate) fn spawn(
        mut command: Command,
        max_message_bytes: usize,
    ) -> Result<Self, std::io::Error> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut child = command.spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("MCP child stdout was unavailable after spawn"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("MCP child stdin was unavailable after spawn"))?;
        Ok(Self {
            child: Some(child),
            transport: BoundedIoTransport::new(stdout, stdin, max_message_bytes),
        })
    }
}

impl Transport<RoleClient> for BoundedChildProcess {
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.transport.send(item)
    }

    fn receive(&mut self) -> impl Future<Output = Option<RxJsonRpcMessage<RoleClient>>> + Send {
        self.transport.receive()
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.transport.close().await?;
        if let Some(mut child) = self.child.take() {
            if child.try_wait()?.is_none() {
                // Initiate termination before the first await so an outer close timeout
                // cannot leave an owned MCP process running in a detached cleanup task.
                if let Err(error) = child.start_kill()
                    && child.try_wait()?.is_none()
                {
                    return Err(error);
                }
            }
            child.wait().await?;
        }
        Ok(())
    }
}

impl Drop for BoundedChildProcess {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.start_kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rmcp::model::{
        CallToolRequestParams, ClientJsonRpcMessage, ClientRequest, PingRequest, RequestId,
    };
    use rmcp::service::ServiceExt;
    use serde_json::{Value, json};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::time::timeout;

    use super::*;

    fn ping() -> ClientJsonRpcMessage {
        ClientJsonRpcMessage::request(
            ClientRequest::PingRequest(PingRequest::default()),
            RequestId::Number(1),
        )
    }

    #[tokio::test]
    async fn outbound_message_is_rejected_before_stdio_write() {
        let (mut peer, transport_side) = tokio::io::duplex(256);
        let (reader, writer) = tokio::io::split(transport_side);
        let mut transport = BoundedIoTransport::new(reader, writer, 16);

        let error = transport.send(ping()).await.unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("exceeded 16 bytes"));
        let mut byte = [0_u8; 1];
        assert!(
            timeout(Duration::from_millis(25), peer.read(&mut byte))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn inbound_raw_limit_fails_an_established_rmcp_call() {
        let (client_side, server_side) = tokio::io::duplex(4096);
        let (reader, writer) = tokio::io::split(client_side);
        let transport = BoundedIoTransport::new(reader, writer, 512);
        let raw_server = tokio::spawn(async move {
            let (reader, mut writer) = tokio::io::split(server_side);
            let mut lines = BufReader::new(reader).lines();
            let initialize: Value = serde_json::from_str(
                &lines
                    .next_line()
                    .await
                    .unwrap()
                    .expect("initialize request"),
            )
            .unwrap();
            assert_eq!(initialize["method"], "initialize");
            let response = json!({
                "jsonrpc": "2.0",
                "id": initialize["id"],
                "result": {
                    "protocolVersion": initialize["params"]["protocolVersion"],
                    "capabilities": {},
                    "serverInfo": { "name": "raw-limit-probe", "version": "1" }
                }
            });
            writer
                .write_all(format!("{response}\n").as_bytes())
                .await
                .unwrap();
            let initialized: Value = serde_json::from_str(
                &lines
                    .next_line()
                    .await
                    .unwrap()
                    .expect("initialized notification"),
            )
            .unwrap();
            assert_eq!(initialized["method"], "notifications/initialized");

            let request: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().expect("tool request"))
                    .unwrap();
            assert_eq!(request["method"], "tools/call");
            let oversized = format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"content\":[]}}}}{}\n",
                request["id"],
                " ".repeat(1024),
            );
            writer.write_all(oversized.as_bytes()).await.unwrap();
        });
        let service = ().serve(transport).await.unwrap();

        let result = timeout(
            Duration::from_secs(1),
            service
                .peer()
                .call_tool(CallToolRequestParams::new("raw_limit_probe")),
        )
        .await
        .expect("oversized response must settle the caller");

        assert!(result.is_err());
        raw_server.await.unwrap();
        let _ = service.cancel().await;
    }
}

use siumai_core::{ApiStability, ModelFamily, NativeSurfaceKind, VerifiedFidelity};
use siumai_provider_alibaba::{
    AlibabaCredential, AlibabaProvider, AlibabaWorkspaceEndpoint, TEXT_EMBEDDING_V3,
    TEXT_EMBEDDING_V4, experimental::AlibabaVideoProviderBuilderExt,
};

#[test]
fn explicit_legacy_endpoints_publish_exact_verified_support_claims() {
    let provider = AlibabaProvider::builder(AlibabaCredential::api_key("test-key"))
        .with_legacy_singapore_language()
        .with_legacy_singapore_embedding()
        .with_legacy_singapore_video()
        .build()
        .unwrap();

    let manifest = provider.support_manifest();
    assert_eq!(manifest.provider_id().as_str(), "alibaba");
    assert_eq!(manifest.profiles().len(), 2);

    let language_claims = manifest.profiles()[0].verified_claims().unwrap();
    assert_eq!(language_claims.len(), 2);
    assert!(
        language_claims
            .iter()
            .all(|claim| claim.scope().family() == ModelFamily::Language)
    );

    let embedding = manifest
        .profiles()
        .iter()
        .find(|profile| {
            profile
                .verified_claims()
                .is_some_and(|claims| claims[0].scope().family() == ModelFamily::Embedding)
        })
        .unwrap();
    let embedding_models = embedding
        .catalog()
        .unwrap()
        .iter()
        .map(|model| model.model().as_str())
        .collect::<Vec<_>>();
    assert_eq!(embedding_models, [TEXT_EMBEDDING_V3, TEXT_EMBEDDING_V4]);

    let video = &manifest.native_claims()[0];
    assert_eq!(video.scope().kind(), NativeSurfaceKind::Job);
    assert_eq!(
        video.scope().binding().surface_id().unwrap().as_str(),
        "video-tasks"
    );
    assert_eq!(video.fidelity(), VerifiedFidelity::Native);
    assert_eq!(video.stability(), ApiStability::Experimental);
}

#[test]
fn workspace_endpoints_never_inherit_legacy_official_claims() {
    let workspace = AlibabaWorkspaceEndpoint::public_origin(
        "https://workspace-id.ap-southeast-1.maas.aliyuncs.com",
    )
    .unwrap();
    let provider = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_language_workspace(&workspace)
        .with_embedding_workspace(&workspace)
        .with_video_workspace(&workspace)
        .build()
        .unwrap();

    let manifest = provider.support_manifest();
    assert_eq!(manifest.profiles().len(), 2);
    assert!(
        manifest
            .profiles()
            .iter()
            .all(|profile| profile.generic_claims().is_some())
    );
    assert!(manifest.native_claims().is_empty());
}

#[test]
fn custom_video_only_configuration_has_an_honest_empty_manifest() {
    let workspace = AlibabaWorkspaceEndpoint::public_origin(
        "https://workspace-id.ap-southeast-1.maas.aliyuncs.com",
    )
    .unwrap();
    let provider = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_video_workspace(&workspace)
        .build()
        .unwrap();

    let manifest = provider.support_manifest();
    assert_eq!(manifest.provider_id().as_str(), "alibaba");
    assert!(manifest.profiles().is_empty());
    assert!(manifest.native_claims().is_empty());
}


from pathlib import Path
import pytest

ARTIFACT = Path(__file__).resolve().parents[2]
ASSET_TESTS = {
    'test_empty_stop_list_recovers_through_turn_markers',
    'test_live_model_config_resolves_roles_and_same_gate',
    'test_actual_gemma_rules_resolve_without_fixed_ids',
}

def pytest_collection_modifyitems(items):
    tokenizers = [
        ARTIFACT / 'models/tokenizers/gemma4_e2b/tokenizer.json',
        ARTIFACT / 'models/derived/gemma4_e2b/embeddings.npy',
    ]
    if all(p.is_file() for p in tokenizers):
        return
    for item in items:
        if item.name in ASSET_TESTS:
            item.add_marker(pytest.mark.skip(reason='Requires the Gemma tokenizer and prepared embeddings (run reconstruction/run.py prepare-assets).'))

"""Downloaded checkpoints stay on disk and are found without network traffic."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import bootstrap
import model_reuse


class ReuseDownloadedModels(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='sb-reuse-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.cfg = copy.deepcopy(bootstrap.DEFAULT_CONFIG)
        self.cfg['engines'] = {e: dict(bootstrap.engine_defaults(e),
            comfy_dir=str(self.root / e), models_dir=str(self.root / e / 'models'))
            for e in bootstrap.ENGINES}
        for e in bootstrap.ENGINES:
            (self.root / e).mkdir()
            (self.root / e / 'main.py').write_text('')
        self.env = mock.patch.dict(os.environ, {'HF_HUB_CACHE': str(self.root / 'hub')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.repo = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
        self.logs = []

    def whole(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'config.json').write_text('{}')
        (path / 'model.safetensors').write_bytes(b'weights')
        return path

    def target(self, repo=None, engine='qwen'):
        return bootstrap.model_dir(bootstrap.engine_models_dir(self.cfg, engine),
                                   repo or self.repo, engine)

    def snapshot(self, repo=None):
        root = model_reuse.hub_cache() / ('models--' + (repo or self.repo).replace('/', '--'))
        (root / 'refs').mkdir(parents=True)
        (root / 'refs' / 'main').write_text('abc123')
        return self.whole(root / 'snapshots' / 'abc123')

    def test_hub_cache_is_linked_to_the_real_qwen_loader_location(self):
        source = self.snapshot()
        with mock.patch.object(bootstrap.requests, 'get', side_effect=AssertionError('network')):
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen', self.logs.append)
        self.assertEqual(self.target().resolve(), source)
        self.assertTrue(bootstrap.model_installed(self.root / 'qwen/models', self.repo))
        self.assertEqual((self.target() / 'model.safetensors').stat().st_ino,
                         (source / 'model.safetensors').stat().st_ino)
        self.assertTrue(self.logs)

    def test_configured_root_reaches_qwen_and_its_hardcoded_tokenizer_check(self):
        shared = self.root / 'external'
        self.cfg['engines']['qwen']['models_dir'] = str(shared)
        tokenizer = 'Qwen/Qwen3-TTS-Tokenizer-12Hz'
        source = self.whole(bootstrap.model_dir(shared, tokenizer))
        self.whole(self.target())
        bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
        actual = bootstrap.model_dir(self.root / 'qwen/models', tokenizer)
        self.assertEqual(actual.resolve(), source)
        self.assertEqual(self.cfg['engines']['qwen']['models_dir'], str(shared))

    def test_moss_existing_named_folder_is_exposed_under_its_flattened_name(self):
        repo = 'OpenMOSS-Team/MOSS-TTS-Local-Transformer'
        source = self.whole(self.root / 'moss/models/moss-tts/MOSS-TTS-Local-Transformer')
        bootstrap.reuse_downloaded_models(self.cfg, 'moss')
        self.assertEqual(self.target(repo, 'moss').resolve(), source)

    def test_extra_model_paths_yaml_uses_relative_base_path(self):
        source = self.whole(self.root / 'shared/Qwen3-TTS-12Hz-0.6B-Base')
        (self.root / 'qwen/extra_model_paths.yaml').write_text('shared:\n  base_path: ..\n  qwen-tts: shared\n')
        bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
        self.assertEqual(self.target().resolve(), source)

    def test_a_broken_alias_is_repaired_without_moving_new_source_weights(self):
        source = self.snapshot()
        target = self.target()
        target.parent.mkdir(parents=True)
        target.symlink_to(self.root / 'old-location', target_is_directory=True)
        bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
        self.assertEqual(target.resolve(), source)
        self.assertTrue((source / 'model.safetensors').is_file())

    def test_a_missing_shard_or_zero_byte_weight_is_not_reused(self):
        source = self.snapshot()
        (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'model.safetensors', 'b': 'second.safetensors'}}))
        self.assertIsNone(model_reuse.cached_snapshot(self.repo))
        (source / 'second.safetensors').write_bytes(b'')
        self.assertIsNone(model_reuse.cached_snapshot(self.repo))
        (source / 'second.safetensors').write_bytes(b'weights')
        self.assertEqual(model_reuse.cached_snapshot(self.repo), source)

    def test_a_nested_tokenizer_is_not_the_parent_checkpoint(self):
        source = self.whole(self.root / 'partial/speech_tokenizer')
        self.assertFalse(model_reuse.whole(source.parent))

    def test_broken_weight_link_and_lfs_pointer_are_not_reused(self):
        source = self.snapshot()
        weight = source / 'model.safetensors'
        weight.unlink()
        weight.symlink_to(self.root / 'missing-blob')
        self.assertFalse(model_reuse.whole(source))
        weight.unlink()
        weight.write_text('version https://git-lfs.github.com/spec/v1\n')
        self.assertFalse(model_reuse.whole(source))

    def test_restarts_do_not_rescan_and_recheck_can_find_new_cache(self):
        with mock.patch.object(model_reuse, 'cached_snapshot', wraps=model_reuse.cached_snapshot) as scan:
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            count = scan.call_count
            # Config survives a process restart; the previous negative result does too.
            self.cfg = json.loads(json.dumps(self.cfg))
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertEqual(scan.call_count, count)
            source = self.snapshot()
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen', force=True)
            self.assertGreater(scan.call_count, count)
            self.assertEqual(self.target().resolve(), source)
            count = scan.call_count
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertEqual(scan.call_count, count)
            shutil.rmtree(source)
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertGreater(scan.call_count, count)
            count = scan.call_count
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertEqual(scan.call_count, count)

    def test_existing_partial_folder_is_kept_and_named_in_the_error(self):
        source = self.snapshot()
        self.target().mkdir(parents=True)
        marker = self.target() / 'model.safetensors.part'
        marker.write_text('unfinished')
        with self.assertRaisesRegex(RuntimeError, 'left unchanged'):
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
        self.assertEqual(marker.read_text(), 'unfinished')
        self.assertTrue(source.is_dir())

    def test_blocked_link_attempt_is_saved_until_recheck_or_location_changes(self):
        self.snapshot()
        self.target().mkdir(parents=True)
        (self.target() / 'model.safetensors.part').write_text('unfinished')
        with mock.patch.object(model_reuse, 'cached_snapshot', wraps=model_reuse.cached_snapshot) as scan:
            with self.assertRaises(RuntimeError):
                bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            count = scan.call_count
            self.cfg = json.loads(json.dumps(self.cfg))
            with self.assertRaises(RuntimeError):
                bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertEqual(scan.call_count, count)
            with self.assertRaises(RuntimeError):
                bootstrap.reuse_downloaded_models(self.cfg, 'qwen', force=True)
            self.assertGreater(scan.call_count, count)

    def test_qwen_tokenizer_alias_failure_is_also_saved_until_recheck(self):
        tokenizer = 'Qwen/Qwen3-TTS-Tokenizer-12Hz'
        self.cfg['engines']['qwen']['models_dir'] = str(self.root / 'shared')
        self.whole(self.target(tokenizer))
        blocked = bootstrap.model_dir(self.root / 'qwen/models', tokenizer)
        blocked.mkdir(parents=True)
        (blocked / 'unfinished.part').write_text('half')
        with mock.patch.object(model_reuse, 'link_directory', wraps=model_reuse.link_directory) as link:
            with self.assertRaises(RuntimeError):
                bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertEqual(link.call_count, 1)
            self.cfg = json.loads(json.dumps(self.cfg))
            with self.assertRaises(RuntimeError):
                bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertEqual(link.call_count, 1)
            with self.assertRaises(RuntimeError):
                bootstrap.reuse_downloaded_models(self.cfg, 'qwen', force=True)
            self.assertEqual(link.call_count, 2)
            shutil.rmtree(blocked)
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertEqual(blocked.resolve(), self.target(tokenizer))

    def test_weights_without_config_are_not_installed(self):
        source = self.whole(self.target())
        (source / 'config.json').unlink()
        self.assertFalse(bootstrap.model_installed(self.root / 'qwen/models', self.repo))

    def test_downloading_an_installed_model_does_not_even_request_a_listing(self):
        self.whole(self.target())
        with mock.patch.object(bootstrap, 'hf_tree', side_effect=AssertionError('network')):
            bootstrap.download_repo(self.cfg, self.repo, self.root / 'qwen/models')

    def test_missing_external_root_is_not_created_and_reappearing_root_resumes(self):
        source = self.snapshot()
        external = self.root / 'offline-drive' / 'sharedmodels'
        self.cfg['engines']['qwen']['models_dir'] = str(external)
        with mock.patch.object(model_reuse, 'cached_snapshot', wraps=model_reuse.cached_snapshot) as scan:
            with self.assertRaisesRegex(RuntimeError, 'location was kept'):
                bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.cfg = json.loads(json.dumps(self.cfg))
            with self.assertRaises(RuntimeError):
                bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            scan.assert_not_called()
            self.assertFalse(external.exists())
            external.mkdir(parents=True)  # the configured drive becomes available
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
            self.assertEqual(self.target().resolve(), source)
            self.assertEqual(self.cfg['engines']['qwen']['models_dir'], str(external))

    def test_explicit_setup_can_create_a_new_custom_destination(self):
        source = self.snapshot()
        new_root = self.root / 'new-model-folder'
        self.cfg['engines']['qwen']['models_dir'] = str(new_root)
        # Automatic startup waits for the missing root; explicit Setup creates it.
        with self.assertRaises(RuntimeError):
            bootstrap.reuse_downloaded_models(self.cfg, 'qwen')
        bootstrap.reuse_downloaded_models(self.cfg, 'qwen', create_root=True)
        self.assertTrue(new_root.is_dir())
        self.assertEqual(self.target().resolve(), source)

    def test_setup_keeps_an_existing_install_and_external_model_root(self):
        shared = self.root / 'external'
        slot = self.cfg['engines']['qwen']
        slot['models_dir'] = str(shared)
        prog = bootstrap.Progress()
        with mock.patch.object(bootstrap, '_run') as run:
            bootstrap._setup_one(self.cfg, prog, 'qwen', 'comfyui', {}, 'python', 'auto', {})
        self.assertEqual(slot['models_dir'], str(shared))
        self.assertEqual(slot['comfy_dir'], str(self.root / 'qwen'))
        self.assertNotIn('clone', run.call_args.args[0])
        self.assertTrue(slot['managed'], 'an existing managed install keeps its environment policy')


class LocationChecksRunOnce(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='sb-locate-once-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.cfg = copy.deepcopy(bootstrap.DEFAULT_CONFIG)
        self.cfg['want_moss'] = False
        self.slot = bootstrap.engine_cfg(self.cfg, 'qwen')
        self.slot['comfy_dir'] = str(self.root / 'gone')

    def test_failed_discovery_is_persistent_and_manual_recheck_retries_once(self):
        with mock.patch.object(bootstrap, 'find_comfy_installs', return_value=[]) as scan, \
                mock.patch.object(bootstrap, 'detect_comfy_dirs', return_value=[]) as quick:
            bootstrap.verify_locations(self.cfg)
            self.assertEqual(scan.call_count, 1)
            self.assertFalse(bootstrap.locations_need_search(self.cfg))
            self.cfg = json.loads(json.dumps(self.cfg))
            for _ in range(3):
                bootstrap.verify_locations(self.cfg)
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(quick.call_count, 1)
            bootstrap.verify_locations(self.cfg, force=True)
            self.assertEqual(scan.call_count, 2)
            bootstrap.engine_cfg(self.cfg, 'qwen')['comfy_dir'] = str(self.root / 'another')
            bootstrap.verify_locations(self.cfg)
            self.assertEqual(scan.call_count, 3)

    def test_valid_saved_locations_do_not_search_until_the_path_fails(self):
        install = self.root / 'verified'
        install.mkdir()
        (install / 'main.py').write_text('')
        self.slot['comfy_dir'] = str(install)
        with mock.patch.object(bootstrap, 'find_comfy_installs', return_value=[]) as scan, \
                mock.patch.object(bootstrap, 'detect_comfy_dirs', return_value=[]) as quick:
            bootstrap.verify_locations(self.cfg)
            bootstrap.verify_locations(json.loads(json.dumps(self.cfg)))
            scan.assert_not_called()
            quick.assert_not_called()
            (install / 'main.py').unlink()
            bootstrap.verify_locations(self.cfg)
            bootstrap.verify_locations(self.cfg)
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(self.slot['comfy_dir'], str(install))

    def test_quick_repair_keeps_explicit_missing_root_beside_empty_stock(self):
        install = self.root / 'ComfyUI'
        install.mkdir()
        (install / 'main.py').write_text('')
        (install / 'models').mkdir()
        self.slot['comfy_dir'] = str(install)
        external = self.root / 'offline-drive/sharedmodels'
        self.slot['models_dir'] = str(external)
        unrelated = self.root / 'app/sharedmodels'
        unrelated.mkdir(parents=True)
        with mock.patch.object(bootstrap, 'APP_DIR', self.root / 'app'):
            bootstrap.verify_locations(self.cfg)
            bootstrap.verify_locations(self.cfg, force=True)
        self.assertEqual(self.slot['models_dir'], str(external))
        self.assertFalse(external.exists())

    def test_full_relocation_preserves_separate_root_and_moves_only_default_root(self):
        install = self.root / 'found/ComfyUI'
        install.mkdir(parents=True)
        (install / 'main.py').write_text('')
        (install / 'models').mkdir()
        for separate in (True, False):
            with self.subTest(separate=separate):
                cfg = copy.deepcopy(self.cfg)
                slot = bootstrap.engine_cfg(cfg, 'qwen')
                old_comfy = str(self.root / 'old/ComfyUI')
                old_models = str(self.root / 'offline-drive/sharedmodels') if separate else old_comfy + '/models'
                slot.update(comfy_dir=old_comfy, models_dir=old_models)
                with mock.patch.object(bootstrap, 'detect_comfy_dirs', return_value=[]), \
                        mock.patch.object(bootstrap, 'rebase_path', return_value=None), \
                        mock.patch.object(bootstrap, 'find_comfy_installs', return_value=[install]) as scan:
                    bootstrap.verify_locations(cfg)
                    bootstrap.verify_locations(json.loads(json.dumps(cfg)))
                self.assertEqual(scan.call_count, 1)
                self.assertEqual(slot['comfy_dir'], str(install))
                self.assertEqual(slot['models_dir'], old_models if separate else str(install / 'models'))

    def test_quick_relocation_follows_only_the_old_engine_default(self):
        app = self.root / 'app'
        install = app / bootstrap.ENGINES['qwen']['dir_name']
        install.mkdir(parents=True)
        (install / 'main.py').write_text('')
        (install / 'models').mkdir()
        old_comfy = r'C:\Old\ComfyUI-Qwen3-TTS'
        self.slot.update(comfy_dir=old_comfy, models_dir=old_comfy + r'\models')
        with mock.patch.object(bootstrap, 'APP_DIR', app), \
                mock.patch.object(bootstrap, 'detect_comfy_dirs', return_value=[]), \
                mock.patch.object(bootstrap, 'find_comfy_installs') as scan:
            bootstrap.verify_locations(self.cfg)
        scan.assert_not_called()
        self.assertEqual(self.slot['models_dir'], str(install / 'models'))


if __name__ == '__main__':
    unittest.main()

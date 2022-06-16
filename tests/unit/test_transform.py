from unittest import TestCase
from unittest.mock import patch, ANY
from gdal_subsetter.transform import HarmonyAdapter

from harmony.message import Message


class TestRecolor(TestCase):

    def test_recolor_on_tif_does_not_call_gdaldem(self):
        test_message = Message({
            "format": {
                "mime": 'tif'
            },
            "sources": [{
                "collection": "fake_collection",
                "variables": []
            }]
        })
        test_adapter = HarmonyAdapter(test_message, '', None)

        src_file = 'sourcefile';
        with patch.object(test_adapter, 'cmd') as mock_cmd:
            result = test_adapter.recolor('granId__subsetted', src_file,
                                          'dstdir')
            self.assertEqual(result, src_file, 'file has changed')
            mock_cmd.assert_not_called()

    @patch('gdal_subsetter.transform.copyfile')
    def test_recolor_on_png_calls_gdaldem(self, copyfile_mock):
        test_message = Message({
            "format": {
                "mime": 'image/png'
            },
            "sources": [{
                "collection": "fake_collection",
                "variables": []
            }]
        })
        test_adapter = HarmonyAdapter(test_message, '', None)

        layer_id = 'granId__subsetted'
        dest_dir = 'destdir'
        expected_outfile = f'{dest_dir}/{layer_id}__colored.png'

        with patch.object(test_adapter, 'cmd') as cmd_mock:
            result = test_adapter.recolor(layer_id, 'srcfile',
                                          dest_dir)
            self.assertEqual(result, expected_outfile,
                             'incorrect output file generated')
            cmd_mock.assert_called_once_with('gdaldem', 'color-relief',
                                             '-alpha', '-of', 'PNG', '-co',
                                             'WORLDFILE=YES', 'srcfile', ANY,
                                             expected_outfile)

            copyfile_mock.assert_any_call(expected_outfile,
                                          f'{dest_dir}/result.png')
            copyfile_mock.assert_any_call(
                expected_outfile.replace('.png', '.wld'), f'{dest_dir}/result.wld')

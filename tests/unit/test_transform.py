from string import ascii_letters, digits
from random import choice
from unittest import TestCase
from unittest.mock import patch, ANY, Mock
from gdal_subsetter.transform import HarmonyAdapter
from gdal_subsetter.exceptions import IncompatibleVariablesError

from harmony.message import Message


def random_file(size=6, chars=ascii_letters + digits):
    return ''.join(choice(chars) for _ in range(size))


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

        expected = 'sourcefile'
        with patch.object(test_adapter, 'cmd') as mock_cmd:
            actual = test_adapter.recolor('granId__subsetted', expected,
                                          'dstdir')
            self.assertEqual(expected, actual, 'file has changed')
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
            actual = test_adapter.recolor(layer_id, 'srcfile', dest_dir)
            self.assertEqual(expected_outfile, actual,
                             'incorrect output file generated')
            cmd_mock.assert_called_once_with('gdaldem', 'color-relief',
                                             '-alpha', '-of', 'PNG', '-co',
                                             'WORLDFILE=YES', 'srcfile', ANY,
                                             expected_outfile)

            copyfile_mock.assert_any_call(expected_outfile,
                                          f'{dest_dir}/result.png')
            copyfile_mock.assert_any_call(
                expected_outfile.replace('.png', '.wld'),
                f'{dest_dir}/result.wld')


class TestAddToList(TestCase):

    def test_add_to_result_stacks_if_stackable_with_netcdf(self):

        filelist = [random_file(), random_file()]
        dstdir = random_file()
        expected = f'{dstdir}/result.tif'

        test_adapter = HarmonyAdapter(
            Message({"format": {
                "mime": "application/x-netcdf4"
            }}), '', None)

        stack_mock = Mock()
        stack_mock.return_value = expected

        checkstackable_mock = Mock()
        checkstackable_mock.return_value = True

        test_adapter.stack_multi_file_with_metadata = stack_mock
        test_adapter.checkstackable = checkstackable_mock

        actual = test_adapter.add_to_result(filelist, dstdir)

        stack_mock.assert_called_once_with(filelist, expected)
        self.assertEqual(expected, actual)

    def test_add_to_result_raises_if_unstackable_with_netcdf(self):

        filelist = [random_file(), random_file()]
        dstdir = random_file()
        dstfile = f'{dstdir}/result.tif'

        test_adapter = HarmonyAdapter(
            Message({"format": {
                "mime": "application/x-netcdf4"
            }}), '', None)

        stack_mock = Mock()
        stack_mock.return_value = dstfile

        checkstackable_mock = Mock()
        checkstackable_mock.return_value = False

        test_adapter.stack_multi_file_with_metadata = stack_mock
        test_adapter.checkstackable = checkstackable_mock

        with self.assertRaises(IncompatibleVariablesError) as error:
            test_adapter.add_to_result(filelist, dstdir)

        stack_mock.assert_not_called()
        self.assertEqual(
            'Request cannot be completed: datasets are incompatible and cannot be combined.',
            str(error.exception)
        )

    def test_add_to_result_stacks_with_png(self):

        filelist = [random_file(), random_file()]
        dstdir = random_file()
        expected = f'{dstdir}/result.png'

        test_adapter = HarmonyAdapter(
            Message({"format": {
                "mime": "png"
            }}), '', None)

        stack_mock = Mock()

        checkstackable_mock = Mock()
        checkstackable_mock.return_value = False

        test_adapter.stack_multi_file_with_metadata = stack_mock
        test_adapter.checkstackable = checkstackable_mock

        actual = test_adapter.add_to_result(filelist, dstdir)

        stack_mock.assert_not_called()
        self.assertEqual(expected, actual)

    def test_add_to_result_stacks_with_jpeg(self):

        filelist = [random_file(), random_file()]
        dstdir = random_file()
        expected = f'{dstdir}/result.jpeg'

        test_adapter = HarmonyAdapter(
            Message({"format": {
                "mime": "jpeg"
            }}), '', None)

        stack_mock = Mock()

        checkstackable_mock = Mock()
        checkstackable_mock.return_value = False

        test_adapter.stack_multi_file_with_metadata = stack_mock
        test_adapter.checkstackable = checkstackable_mock

        actual = test_adapter.add_to_result(filelist, dstdir)

        stack_mock.assert_not_called()
        self.assertEqual(expected, actual)

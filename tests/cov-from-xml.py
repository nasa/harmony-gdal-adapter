
import sys
import xml.etree.ElementTree as ET


def print_cov_percentage(cov_xml):
    tree = ET.parse(cov_xml)
    root = tree.getroot()

    cov_rate = round(float(root.attrib['line-rate']), 2) * 100
    print(int(cov_rate))


if __name__ == "__main__":
    cov_xml_file = sys.argv[1]

    print_cov_percentage(cov_xml_file)

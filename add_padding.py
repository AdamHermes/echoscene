import sys
import xml.etree.ElementTree as ET

def add_padding_to_svg(filepath, pad_pt=10):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Parse viewBox
    if 'viewBox' in root.attrib:
        viewbox = root.attrib['viewBox'].split()
        if len(viewbox) == 4:
            x, y, w, h = map(float, viewbox)
            # Add padding
            x -= pad_pt
            y -= pad_pt
            w += 2 * pad_pt
            h += 2 * pad_pt
            root.attrib['viewBox'] = f"{x} {y} {w} {h}"
            
            # Update width and height if they exist and are in pt or px
            for attr in ['width', 'height']:
                if attr in root.attrib:
                    val_str = root.attrib[attr]
                    if val_str.endswith('pt'):
                        val = float(val_str[:-2])
                        root.attrib[attr] = f"{val + 2 * pad_pt}pt"
                    elif val_str.endswith('px'):
                        val = float(val_str[:-2])
                        root.attrib[attr] = f"{val + 2 * pad_pt}px"
                    else:
                        try:
                            val = float(val_str)
                            root.attrib[attr] = str(val + 2 * pad_pt)
                        except ValueError:
                            pass
            
            tree.write(filepath, xml_declaration=True, encoding='utf-8')
            print(f"Added {pad_pt}pt padding to {filepath}")
        else:
            print(f"Invalid viewBox in {filepath}")
    else:
        print(f"No viewBox found in {filepath}")

if __name__ == "__main__":
    for f in sys.argv[1:]:
        add_padding_to_svg(f, pad_pt=-60)

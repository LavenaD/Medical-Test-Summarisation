
import xml.etree.ElementTree as ET

class XmlReader:
    ROOT_ELEMENT = ".//pmcId"
    NODE_ELEMENTS = [".//MedlineCitation/Article/Abstract/AbstractText"]
    # FIELDNAMES = {"ID" : "id", "COMPARISON" : "comparison", "INDICATION" : "indication", "FINDINGS" : "findings", "IMPRESSION" : "impression"}
    FIELDNAMES = {"FINDINGS" : "findings", "IMPRESSION" : "impression"}
    def __init__(self ):
        self.data_row = {}
        
        for fieldname in XmlReader.FIELDNAMES.keys():
            self.data_row[XmlReader.FIELDNAMES[fieldname]] = None

    def __read_id(self):
        pcm_code = self.root.find(XmlReader.ROOT_ELEMENT)
        if pcm_code is not None:
            self.data_row[XmlReader.FIELDNAMES["ID"]] = pcm_code.get("id").strip()
        else:
            self.data_row[XmlReader.FIELDNAMES["ID"]] = "ID NOT FOUND"
        return self.data_row.get(XmlReader.FIELDNAMES["ID"])

    def __read_abstract_text(self):
        for node_element in XmlReader.NODE_ELEMENTS:
            if self.root.findall(node_element) is not None:
                for innerchild in self.root.findall(node_element):
                    label = innerchild.get("Label")
                    value = innerchild.text
                    if label in XmlReader.FIELDNAMES.keys():
                        self.data_row[XmlReader.FIELDNAMES[label]] = value
        if self.data_row.get(XmlReader.FIELDNAMES["FINDINGS"]) is None:
            return None
        return self.data_row
    
    def read_file(self, file_path):
        self.file_path = file_path
        tree = ET.parse(self.file_path)
        self.root = tree.getroot()
        # pcm_id = self.__read_id()
        # if pcm_id is None:
        #     # write to log file
        #     return None
        data_dict = self.__read_abstract_text()
        return data_dict
    
    
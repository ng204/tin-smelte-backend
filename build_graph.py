# -*- coding: UTF-8 -*-
import os

from py2neo import Graph, Node, Relationship
import config
import pandas as pd


class TinSmelteGraph():
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data/锡冶炼总三元组.xlsx')
        self.g = Graph(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))

    def build(self):
        df = pd.read_excel(self.data_path, sheet_name='总三元组（旧）')
        for _, row in df.iterrows():
            start_node = Node(row[3], name=row[0])
            self.g.merge(start_node, row[3], 'name')
            end_node = Node(row[3], name=row[2])
            self.g.merge(end_node, row[3], 'name')
            relationship_type = row[1]
            relationship = Relationship(start_node, relationship_type, end_node)
            self.g.merge(relationship)
        df2 = pd.read_excel(self.data_path, sheet_name='参数与模式层')
        for _, row in df2.iterrows():
            start_node = Node(row[3], name=row[0])
            self.g.merge(start_node, row[3], 'name')
            end_node = Node(row[3], name=row[2])
            self.g.merge(end_node, row[3], 'name')
            relationship_type = row[1]
            relationship = Relationship(start_node, relationship_type, end_node)
            self.g.merge(relationship)
        df3 = pd.read_excel(self.data_path, sheet_name='总三元组（新）')
        for _, row in df3.iterrows():
            start_node = Node(row[3], name=row[0])
            self.g.merge(start_node, row[3], 'name')
            end_node = Node(row[3], name=row[2])
            self.g.merge(end_node, row[3], 'name')
            relationship_type = row[1]
            relationship = Relationship(start_node, relationship_type, end_node)
            self.g.merge(relationship)


if __name__ == '__main__':
    t = TinSmelteGraph()
    t.build()

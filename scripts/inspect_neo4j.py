import sys
sys.path.append('..')
import config
from py2neo import Graph

def inspect(name):
    g = Graph(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    print('Connected to Neo4j at', config.NEO4J_URI)
    q = "MATCH (n {name:$name})-[r]->(m) RETURN n, r, m, type(r) as relType LIMIT 100"
    rows = g.run(q, name=name).data()
    if not rows:
        print('No exact-match outgoing relations, trying contains...')
        q2 = "MATCH (n) WHERE (n.name IS NOT NULL AND n.name CONTAINS $name) OR (n.nodename IS NOT NULL AND n.nodename CONTAINS $name) WITH n LIMIT 5 MATCH (n)-[r]->(m) RETURN n, r, m, type(r) as relType LIMIT 200"
        rows = g.run(q2, name=name).data()

    print(f'Found {len(rows)} rows')
    for i,row in enumerate(rows,1):
        print('--- ROW', i, '---')
        try:
            n = row.get('n')
            m = row.get('m')
            r = row.get('r')
            relType = row.get('relType')
            print('n dict:', dict(n) if n is not None else None)
            print('m dict:', dict(m) if m is not None else None)
            print('relType from cypher:', relType)
            print('r repr:', repr(r))
            # attempt to show attributes
            try:
                print('r type attr:', getattr(r, 'type', None))
            except Exception as e:
                print('r.type access error:', e)
        except Exception as e:
            print('row inspect error:', e)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python inspect_neo4j.py "节点名"')
        sys.exit(1)
    inspect(sys.argv[1])











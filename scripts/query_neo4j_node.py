import sys, json
sys.path.append('..')
import config
from py2neo import Graph

def query(name):
    g = Graph(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    # find exact match and contains
    q = "MATCH (n) WHERE (n.name IS NOT NULL AND n.name = $name) OR (n.nodename IS NOT NULL AND n.nodename = $name) RETURN n LIMIT 5"
    exact = g.run(q, name=name).data()
    if not exact:
        # try contains
        q2 = "MATCH (n) WHERE (n.name IS NOT NULL AND n.name CONTAINS $name) OR (n.nodename IS NOT NULL AND n.nodename CONTAINS $name) RETURN n LIMIT 20"
        exact = g.run(q2, name=name).data()

    # get outgoing relations
    q3 = "MATCH (n {name: $name})-[r]->(m) RETURN n, r, m LIMIT 50"
    out = g.run(q3, name=name).data()
    # map to simple dicts
    out_simple = []
    for row in out:
        try:
            n = dict(row.get('n') or {})
            m = dict(row.get('m') or {})
            r = row.get('r')
            # try to extract relation type
            rel_type = None
            try:
                rel_type = g.run("RETURN type($r)", r=r).evaluate()
            except Exception:
                pass
            # fallback
            if not rel_type:
                try:
                    rel_type = getattr(r, 'type', None)
                except Exception:
                    rel_type = None
            out_simple.append({
                'start': n.get('name') or n.get('nodename'),
                'end': m.get('name') or m.get('nodename'),
                'rel_type': rel_type,
                'raw_r': str(r),
                'r_repr': repr(r)
            })
        except Exception as e:
            out_simple.append({'error': str(e)})

    # convert any py2neo objects to simple dicts/strings for printing
    def serialize_row(r):
        try:
            if hasattr(r, 'items'):
                return dict(r)
            return str(r)
        except Exception:
            return str(r)

    ser_exact = [serialize_row(x) for x in exact]
    print(json.dumps({'exact_nodes': ser_exact, 'out': out_simple}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python query_neo4j_node.py "节点名"')
        sys.exit(1)
    name = sys.argv[1]
    query(name)



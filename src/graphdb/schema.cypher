// Neo4j schema. Owner: Member 4 (feature/graph-tool).
//
// Relationships (the arrow direction matters!):
//   (Course)-[:BELONGS_TO]->(Subject)        a course belongs to a subject
//   (Course)-[:HAS_TOPIC]->(Topic)           a course teaches a topic
//   (Course)-[:PREREQUISITE_OF]->(Course)    A-[:PREREQUISITE_OF]->B  means: take A BEFORE B
//   (Topic)-[:REQUIRES]->(Topic)             A-[:REQUIRES]->B         means: learn B BEFORE A
//   (Resource)-[:COVERS]->(Topic)            a resource teaches a topic
//   (Topic)-[:BEST_STUDIED_WITH]->(Strategy) a topic is best learned with a strategy
//
// Node properties:
//   Course(name, code)   Subject(name)   Topic(name, description)
//   Resource(title, url, type)   Strategy(name, description)
//
// NOTE: Topic and Strategy names must match the names used in data/documents/ (agreed with Member 2).
//
// The statements below are separated by semicolons. seed_graph.py runs them one by one.
// Lines starting with // are comments and are ignored.

// Uniqueness constraints: no two nodes of the same label can share the same name.
// This also creates an index on that property, which makes lookups fast.
CREATE CONSTRAINT course_name IF NOT EXISTS FOR (c:Course) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT subject_name IF NOT EXISTS FOR (s:Subject) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT resource_title IF NOT EXISTS FOR (r:Resource) REQUIRE r.title IS UNIQUE;
CREATE CONSTRAINT strategy_name IF NOT EXISTS FOR (st:Strategy) REQUIRE st.name IS UNIQUE;

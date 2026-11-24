// Neo4j schema (starting point). Owner: Member 4 (feature/graph-tool).
//
// Relationships:
//   (Course)-[:BELONGS_TO]->(Subject)
//   (Course)-[:HAS_TOPIC]->(Topic)
//   (Course)-[:PREREQUISITE_OF]->(Course)
//   (Topic)-[:REQUIRES]->(Topic)
//   (Resource)-[:COVERS]->(Topic)
//   (Topic)-[:BEST_STUDIED_WITH]->(Strategy)
//
// NOTE: Topic names must match the topic names used in data/documents/ (agree with Member 2).

// TODO (Member 4): uniqueness constraints (Neo4j 5 syntax)
// CREATE CONSTRAINT course_name IF NOT EXISTS FOR (c:Course) REQUIRE c.name IS UNIQUE;
// CREATE CONSTRAINT subject_name IF NOT EXISTS FOR (s:Subject) REQUIRE s.name IS UNIQUE;
// CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE;
// CREATE CONSTRAINT resource_title IF NOT EXISTS FOR (r:Resource) REQUIRE r.title IS UNIQUE;
// CREATE CONSTRAINT strategy_name IF NOT EXISTS FOR (st:Strategy) REQUIRE st.name IS UNIQUE;

// TODO (Member 4): node properties
//   Course(name, code?), Subject(name), Topic(name, description?),
//   Resource(title, url, type?), Strategy(name, description?)
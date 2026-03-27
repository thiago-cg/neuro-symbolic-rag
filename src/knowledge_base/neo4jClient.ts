import neo4j, { type Driver, type Session } from "neo4j-driver";
import type { Config } from "../config.js";
import type { Paper, Triple } from "../graph/state.js";
import { getLogger } from "../observability.js";

const log = getLogger("neo4j");

export class Neo4jClient {
  private driver: Driver | undefined;

  constructor(private readonly cfg: Config) {}

  async connect(): Promise<void> {
    this.driver = neo4j.driver(
      this.cfg.neo4jUri,
      neo4j.auth.basic(this.cfg.neo4jUser, this.cfg.neo4jPassword),
    );
    await this.driver.verifyConnectivity();
    log.info({ uri: this.cfg.neo4jUri }, "Neo4j connected");
    await this.initSchema();
  }

  async disconnect(): Promise<void> {
    await this.driver?.close();
    this.driver = undefined;
  }

  private getDriver(): Driver {
    if (!this.driver) throw new Error("Neo4j not connected — call connect() first");
    return this.driver;
  }

  private session(): Session {
    return this.getDriver().session();
  }

  private async initSchema(): Promise<void> {
    const s = this.session();
    try {
      await s.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paperId IS UNIQUE");
      await s.run("CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)");
      await s.run("CREATE INDEX triple_predicate IF NOT EXISTS FOR ()-[r:TRIPLE]-() ON (r.predicate)");
      log.debug("Neo4j schema initialized");
    } finally {
      await s.close();
    }
  }

  async resetSchema(): Promise<void> {
    const s = this.session();
    try {
      await s.run("MATCH (n) DETACH DELETE n");
      await s.run("DROP CONSTRAINT paper_id IF EXISTS");
      await s.run("DROP INDEX paper_title IF EXISTS");
      await s.run("DROP INDEX triple_predicate IF EXISTS");
      log.info("Neo4j schema reset");
    } finally {
      await s.close();
    }
    await this.initSchema();
  }

  async insertPapers(papers: Paper[]): Promise<void> {
    if (papers.length === 0) return;
    const s = this.session();
    try {
      await s.run(
        `UNWIND $papers AS p
         MERGE (n:Paper { paperId: p.paperId })
         SET n.title = p.title,
             n.abstract = p.abstract,
             n.year = p.year,
             n.url = p.url,
             n.authors = p.authors`,
        { papers },
      );
      log.debug({ count: papers.length }, "Inserted papers");
    } finally {
      await s.close();
    }
  }

  async insertTriples(triples: Triple[]): Promise<void> {
    if (triples.length === 0) return;
    const s = this.session();
    try {
      await s.run(
        `UNWIND $triples AS t
         MERGE (a:Entity { name: t.subject })
         MERGE (b:Entity { name: t.object })
         CREATE (a)-[:TRIPLE {
           predicate: t.predicate,
           confidence: t.confidence,
           source: t.source
         }]->(b)`,
        { triples },
      );
      log.debug({ count: triples.length }, "Inserted triples");
    } finally {
      await s.close();
    }
  }

  async insertInferredFacts(facts: string[], source: string): Promise<void> {
    if (facts.length === 0) return;
    const s = this.session();
    try {
      await s.run(
        `UNWIND $facts AS f
         CREATE (n:InferredFact { text: f, source: $source, createdAt: datetime() })`,
        { facts, source },
      );
    } finally {
      await s.close();
    }
  }

  async query(cypher: string): Promise<Record<string, unknown>[]> {
    const writeKeywords = ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"];
    const upper = cypher.toUpperCase();
    if (writeKeywords.some((kw) => upper.includes(kw))) {
      throw new Error("Write operations are not allowed via the query endpoint");
    }

    const s = this.getDriver().session({ defaultAccessMode: neo4j.session.READ });
    try {
      const result = await s.run(cypher);
      return result.records.map((r) => r.toObject());
    } finally {
      await s.close();
    }
  }
}

import { readFileSync } from "fs";
import initSqlJs from "sql.js";

export interface SqliteData {
  tables: string[];
  tableTexts: Record<string, string>;
  text: string;
}

export async function parseSqlite(filePath: string): Promise<SqliteData> {
  const SQL = await initSqlJs();
  const buffer = readFileSync(filePath);
  const db = new SQL.Database(buffer);

  const tablesResult = db.exec(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
  );
  const tables: string[] =
    tablesResult[0]?.values.map((row) => row[0] as string) ?? [];

  const tableTexts: Record<string, string> = {};

  for (const table of tables) {
    const result = db.exec(`SELECT * FROM "${table}" LIMIT 100`);
    if (!result[0]) continue;

    const { columns, values } = result[0];
    const lines = [
      `Table: ${table}`,
      `Columns: ${columns.join(", ")}`,
      ...values.map((row) =>
        columns.map((col, i) => `${col}=${row[i]}`).join(" | "),
      ),
    ];
    tableTexts[table] = lines.join("\n");
  }

  db.close();

  return {
    tables,
    tableTexts,
    text: Object.values(tableTexts).join("\n\n"),
  };
}

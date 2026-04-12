import { sql } from "@vercel/postgres";
import fs from "fs";
import bcrypt from "bcrypt";
import path from "path";

async function seed() {
  console.log("Loading demoData.json...");
  const dataPath = path.resolve("./demoData.json");
  const rawData = fs.readFileSync(dataPath, "utf-8");
  const { user, tasks } = JSON.parse(rawData);

  console.log("Creating/Ensuring users and tasks tables exist...");
  await sql`
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      name VARCHAR(255) NOT NULL,
      email VARCHAR(255) UNIQUE NOT NULL,
      password VARCHAR(255) NOT NULL,
      timezone VARCHAR(255),
      role INT,
      wake_time INT,
      sleep_time INT,
      session_style INT,
      switch_buffer INT,
      deadline_style INT,
      peak_focus_windows JSONB DEFAULT '[]'::jsonb,
      low_energy_windows JSONB DEFAULT '[]'::jsonb,
      fixed_commitments JSONB DEFAULT '[]'::jsonb,
      hard_exclusions JSONB DEFAULT '[]'::jsonb,
      recovery_activities JSONB DEFAULT '[]'::jsonb,
      created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
  `;

  await sql`
    CREATE TABLE IF NOT EXISTS tasks (
        id VARCHAR(255) PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        type VARCHAR(50) NOT NULL,
        difficulty INTEGER NOT NULL,
        duration INTEGER NOT NULL,
        priority VARCHAR(20) NOT NULL,
        state VARCHAR(50) NOT NULL,
        subject VARCHAR(255),
        deadline TIMESTAMPTZ,
        energy_recovery FLOAT,
        cl FLOAT NOT NULL,
        cl_breakdown JSONB NOT NULL,
        scheduled_slot JSONB,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
  `;

  try {
    await sql`ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_user_id_fkey;`;
    await sql`ALTER TABLE tasks ADD CONSTRAINT tasks_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;`;
  } catch (e) {}

  console.log("Deleting existing demo user and their tasks...");
  await sql`DELETE FROM users WHERE email = ${user.email}`;

  console.log("Hashing password...");
  const hashedPassword = await bcrypt.hash(user.password, 10);

  console.log("Inserting demo user...");
  const insertedUser = await sql`
      INSERT INTO users (
        name, email, password, timezone, role, wake_time, sleep_time, session_style, switch_buffer, deadline_style,
        peak_focus_windows, low_energy_windows, fixed_commitments, hard_exclusions, recovery_activities
      )
      VALUES (
        ${user.name}, ${user.email}, ${hashedPassword}, ${user.timezone}, ${user.role}, ${user.wake_time}, ${user.sleep_time},
        ${user.session_style}, ${user.switch_buffer}, ${user.deadline_style},
        ${JSON.stringify(user.peak_focus_windows)}::jsonb,
        ${JSON.stringify(user.low_energy_windows)}::jsonb,
        ${JSON.stringify(user.fixed_commitments)}::jsonb,
        ${JSON.stringify(user.hard_exclusions)}::jsonb,
        ${JSON.stringify(user.recovery_activities)}::jsonb
      )
      RETURNING id;
  `;

  const userId = insertedUser.rows[0].id;

  console.log(`User inserted with ID: ${userId}. Inserting tasks...`);
  for (const task of tasks) {
    await sql`
      INSERT INTO tasks (
        id, user_id, name, type, difficulty, duration, priority, state, subject,
        deadline, energy_recovery, cl, cl_breakdown, scheduled_slot, created_at
      ) VALUES (
        ${task.id}, ${userId}, ${task.name}, ${task.type}, ${task.difficulty}, ${task.duration},
        ${task.priority}, ${task.state}, ${task.subject || null},
        ${task.deadline ? new Date(task.deadline).toISOString() : null}, ${task.energy_recovery || null}, ${task.cl},
        ${JSON.stringify(task.cl_breakdown)}::jsonb,
        ${task.scheduled_slot ? JSON.stringify(task.scheduled_slot) : null}::jsonb,
        ${task.created_at}
      );
    `;
  }

  console.log("Demo user seeded successfully!");
  process.exit(0);
}

seed().catch((err) => {
  console.error(err);
  process.exit(1);
});

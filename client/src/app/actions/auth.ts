"use server";

import { sql } from "@vercel/postgres";
import bcrypt from "bcrypt";
import { sign } from "jsonwebtoken";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";

type TimeWindow = { start_min: number; end_min: number };
type FixedBlock = {
  title: string;
  start_min: number;
  end_min: number;
  days: number[];
  recurring: boolean;
};
type TimeExclusion = {
  label: string;
  start_min: number;
  end_min: number;
  days: number[];
};
type RecoveryActivity = {
  name: string;
  duration_min: number;
  energy_value: number;
};

type RegisterUserInput = {
  name: string;
  email: string;
  password: string;
  timezone: string;
  role: number;
  wake_sleep: {
    wake: number;
    sleep: number;
  };
  session_config: {
    session_style: number;
    switch_buffer: number;
  };
  deadline_style: number;
  peak_focus_windows?: TimeWindow[];
  low_energy_windows?: TimeWindow[];
  fixed_commitments?: FixedBlock[];
  hard_exclusions?: TimeExclusion[];
  recovery_activities?: RecoveryActivity[];
};

type LoginUserInput = {
  email: string;
  password: string;
};

type ActionResult = {
  error?: string;
};

async function authenticateAndSetSession(
  credentials: LoginUserInput,
): Promise<ActionResult> {
  const { email, password } = credentials;

  try {
    const result = await sql`
      SELECT id, password FROM users WHERE email = ${email} LIMIT 1;
    `;

    if (result.rows.length === 0) {
      return { error: "User not found." };
    }

    const user = result.rows[0] as { id: number; password: string };
    const passwordMatch = await bcrypt.compare(password, user.password);

    if (!passwordMatch) {
      return { error: "Invalid password." };
    }

    if (!process.env.JWT_SECRET) {
      return { error: "Server is missing JWT_SECRET." };
    }

    const token = sign({ userId: user.id }, process.env.JWT_SECRET, {
      expiresIn: "1h",
    });

    const cookieStore = await cookies();
    cookieStore.set("token", token, {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      path: "/",
      maxAge: 60 * 60,
    });

    return {};
  } catch (error) {
    console.error("Login failed:", error);
    return { error: "Login failed. Please try again." };
  }
}

export async function createUsersTable(): Promise<ActionResult> {
  try {
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
        user_id INTEGER REFERENCES users(id),
        name VARCHAR(255) NOT NULL,
        type VARCHAR(50) NOT NULL,
        difficulty INTEGER NOT NULL,
        completion_time INTEGER NOT NULL,
        priority VARCHAR(20) NOT NULL,
        state VARCHAR(50) NOT NULL,
        subject VARCHAR(255),
        deadline TIMESTAMPTZ,
        energy_recovery FLOAT,
        etask FLOAT NOT NULL,
        cl_breakdown JSONB NOT NULL,
        scheduled_slot JSONB,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
      );
    `;

    await sql`
      CREATE TABLE IF NOT EXISTS section_weights (
        user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        morning DOUBLE PRECISION NOT NULL DEFAULT 0.40,
        afternoon DOUBLE PRECISION NOT NULL DEFAULT 0.35,
        evening DOUBLE PRECISION NOT NULL DEFAULT 0.25,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
      );
    `;
    // Add columns if they don't exist (for existing tables)
    await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS peak_focus_windows JSONB DEFAULT '[]'::jsonb`;
    await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS low_energy_windows JSONB DEFAULT '[]'::jsonb`;
    await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS fixed_commitments JSONB DEFAULT '[]'::jsonb`;
    await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS hard_exclusions JSONB DEFAULT '[]'::jsonb`;
    await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS recovery_activities JSONB DEFAULT '[]'::jsonb`;
    await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS experience_level VARCHAR(20) DEFAULT 'beginner'`;
    await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS learning_style VARCHAR(20) DEFAULT 'balanced'`;
    return {};
  } catch (error) {
    console.error("Create users table failed:", error);
    return { error: "Failed to create users table." };
  }
}

export async function registerUser(
  userData: RegisterUserInput,
): Promise<ActionResult> {
  const {
    name,
    email,
    password,
    timezone,
    role,
    wake_sleep,
    session_config,
    deadline_style,
    peak_focus_windows = [],
    low_energy_windows = [],
    fixed_commitments = [],
    hard_exclusions = [],
    recovery_activities = [],
  } = userData;

  try {
    // Ensure schema is up to date
    await createUsersTable();

    const existing = await sql`
      SELECT id FROM users WHERE email = ${email} LIMIT 1;
    `;

    if (existing.rows.length > 0) {
      return { error: "Email is already registered." };
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const peakJson = JSON.stringify(peak_focus_windows);
    const lowJson = JSON.stringify(low_energy_windows);
    const fixedJson = JSON.stringify(fixed_commitments);
    const exclusionsJson = JSON.stringify(hard_exclusions);
    const recoveryJson = JSON.stringify(recovery_activities);

    const inserted = await sql`
      INSERT INTO users (
        name,
        email,
        password,
        timezone,
        role,
        wake_time,
        sleep_time,
        session_style,
        switch_buffer,
        deadline_style,
        peak_focus_windows,
        low_energy_windows,
        fixed_commitments,
        hard_exclusions,
        recovery_activities
      )
      VALUES (
        ${name},
        ${email},
        ${hashedPassword},
        ${timezone},
        ${role},
        ${wake_sleep.wake},
        ${wake_sleep.sleep},
        ${session_config.session_style},
        ${session_config.switch_buffer},
        ${deadline_style},
        ${peakJson}::jsonb,
        ${lowJson}::jsonb,
        ${fixedJson}::jsonb,
        ${exclusionsJson}::jsonb,
        ${recoveryJson}::jsonb
      )
      RETURNING id;
    `;

    const newUserId = inserted.rows[0]?.id as number | undefined;
    if (!newUserId) {
      return { error: "Registration failed while creating user profile." };
    }

    await sql`
      INSERT INTO section_weights (user_id, morning, afternoon, evening)
      VALUES (${newUserId}, 0.40, 0.35, 0.25)
      ON CONFLICT (user_id) DO NOTHING;
    `;
  } catch (error) {
    console.error("Registration failed:", error);
    return { error: "Registration failed. Please try again." };
  }

  return authenticateAndSetSession({ email, password });
}

export async function loginUser(
  formData: FormData,
): Promise<ActionResult | void> {
  const email = String(formData.get("email") ?? "");
  const password = String(formData.get("password") ?? "");

  const authResult = await authenticateAndSetSession({ email, password });
  if (authResult.error) {
    return authResult;
  }

  redirect("/dashboard");
}

export async function getUserProfile() {
  const cookieStore = await cookies();
  const token = cookieStore.get("token")?.value;
  if (!token) return { error: "Not authenticated" };

  try {
    if (!process.env.JWT_SECRET) {
      return { error: "Server is missing JWT_SECRET." };
    }

    // Ensure JSONB columns exist on older tables
    await createUsersTable();

    const { verify } = await import("jsonwebtoken");
    const decoded = verify(token, process.env.JWT_SECRET) as { userId: number };

    const result = await sql`
      SELECT id, name, email, timezone, role, wake_time, sleep_time, session_style, switch_buffer, deadline_style,
        peak_focus_windows, low_energy_windows, fixed_commitments, hard_exclusions, recovery_activities,
        experience_level, learning_style, created_at
      FROM users
      WHERE id = ${decoded.userId}
      LIMIT 1;
    `;

    if (result.rows.length === 0) {
      return { error: "User not found" };
    }

    return { user: result.rows[0] as any };
  } catch (error) {
    console.error("Get user profile failed:", error);
    return { error: "Failed to get user profile" };
  }
}

export async function updateUserProfile(updates: {
  wake_time?: number;
  sleep_time?: number;
  peak_focus_windows?: TimeWindow[];
  low_energy_windows?: TimeWindow[];
  fixed_commitments?: FixedBlock[];
  experience_level?: "beginner" | "intermediate" | "advanced";
  learning_style?: "visual" | "reading" | "practice" | "balanced";
}): Promise<ActionResult> {
  const cookieStore = await cookies();
  const token = cookieStore.get("token")?.value;
  if (!token) return { error: "Not authenticated" };

  try {
    if (!process.env.JWT_SECRET) {
      return { error: "Server is missing JWT_SECRET." };
    }
    const { verify } = await import("jsonwebtoken");
    const decoded = verify(token, process.env.JWT_SECRET) as { userId: number };

    if (updates.wake_time !== undefined && updates.sleep_time !== undefined) {
      await sql`
        UPDATE users SET wake_time = ${updates.wake_time}, sleep_time = ${updates.sleep_time}
        WHERE id = ${decoded.userId};
      `;
    }

    if (updates.peak_focus_windows !== undefined) {
      const json = JSON.stringify(updates.peak_focus_windows);
      await sql`UPDATE users SET peak_focus_windows = ${json}::jsonb WHERE id = ${decoded.userId};`;
    }

    if (updates.low_energy_windows !== undefined) {
      const json = JSON.stringify(updates.low_energy_windows);
      await sql`UPDATE users SET low_energy_windows = ${json}::jsonb WHERE id = ${decoded.userId};`;
    }

    if (updates.fixed_commitments !== undefined) {
      const json = JSON.stringify(updates.fixed_commitments);
      await sql`UPDATE users SET fixed_commitments = ${json}::jsonb WHERE id = ${decoded.userId};`;
      // Also update hard_exclusions to match
      const exclusions = updates.fixed_commitments.map((c) => ({
        label: c.title,
        start_min: c.start_min,
        end_min: c.end_min,
        days: c.days,
      }));
      const exJson = JSON.stringify(exclusions);
      await sql`UPDATE users SET hard_exclusions = ${exJson}::jsonb WHERE id = ${decoded.userId};`;
    }

    if (updates.experience_level !== undefined) {
      await sql`UPDATE users SET experience_level = ${updates.experience_level} WHERE id = ${decoded.userId};`;
    }

    if (updates.learning_style !== undefined) {
      await sql`UPDATE users SET learning_style = ${updates.learning_style} WHERE id = ${decoded.userId};`;
    }

    return {};
  } catch (error) {
    console.error("Update profile failed:", error);
    return { error: "Failed to update profile." };
  }
}

export async function logoutUser() {
  const cookieStore = await cookies();
  cookieStore.delete("token");
  redirect("/login");
}

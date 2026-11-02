# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 6 — Background Learning Agent
# ══════════════════════════════════════════════════════════════════
# Phase 10 of the Taseneem Adaptive Coordination Intelligence
# framework: Feedback Intelligence.
#
# What Agent 6 does:
#   1. Stores every completed pipeline case in SQLite
#   2. When a confirmed outcome is submitted (biopsy result,
#      specialist verdict), updates agent trust scores using
#      exponential moving average over recent confirmed cases
#   3. Returns updated trust scores to Orchestrator so future
#      cases use the most current reliability estimates
#
# Never communicates with any other agent directly.
# All communication is exclusively through the Orchestrator.
#
# Database: data/agent6/cases.db
# Tables  : cases, outcomes, trust_history
# ══════════════════════════════════════════════════════════════════

import os
import sys
import sqlite3
import traceback
from datetime import datetime

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ── Config ────────────────────────────────────────────────────────
DB_DIR    = os.path.join(PROJECT_ROOT, "data", "agent6")
DB_PATH   = os.path.join(DB_DIR, "cases.db")
os.makedirs(DB_DIR, exist_ok=True)

# EMA learning rate — conservative to prevent rapid oscillation
ALPHA       = 0.10

# Rolling window for trust computation
WINDOW_SIZE = 50

# Minimum confirmed cases before trust updates are applied
MIN_CASES   = 5

# Trust score bounds
TRUST_FLOOR   = 0.30
TRUST_CEILING = 0.98

# Malignant classes — must match Orchestrator
MALIGNANT_CLASSES = {0, 2, 3}


class BackgroundLearningAgent:
    """
    Phase 10 — Feedback Intelligence.

    Stores completed cases and updates agent trust scores
    from confirmed clinical outcomes. Runs silently in the
    background. Called by Orchestrator only.
    """

    def __init__(self):
        self.name = "Agent 6 — Background Learning Agent"
        print(f"  [{self.name}] Initialising ...")
        self._init_db()
        print(f"  [{self.name}] Database: {DB_PATH}")
        print(f"  [{self.name}] Ready")

    # ── Database initialisation ───────────────────────────────────

    def _init_db(self):
        """
        Initialise SQLite database with three tables.
        Safe to call multiple times — uses CREATE IF NOT EXISTS.
        """
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()

        # cases — every completed pipeline run
        c.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                case_id           TEXT PRIMARY KEY,
                timestamp         TEXT NOT NULL,
                image_path        TEXT,
                predicted_name    TEXT,
                predicted_index   INTEGER,
                confidence        REAL,
                is_malignant      INTEGER,
                is_pox            INTEGER,
                pox_detected      INTEGER,
                decision          TEXT,
                urgency           TEXT,
                conflict_type     TEXT,
                conflict_severity TEXT,
                fused_malignancy  REAL,
                agent1_conf       REAL,
                agent1_trust      REAL,
                agent2_conf       REAL,
                agent2_trust      REAL,
                agent3_conf       REAL,
                agent3_trust      REAL,
                agent4_conf       REAL,
                agent4_trust      REAL,
                clinical_conf     REAL,
                clinical_trust    REAL,
                agent5_conf       REAL,
                has_symptoms      INTEGER,
                risk_level        TEXT,
                risk_score        INTEGER,
                pipeline_time_s   REAL
            )
        ''')

        # outcomes — confirmed clinical results
        c.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                outcome_id          TEXT PRIMARY KEY,
                case_id             TEXT NOT NULL,
                confirmed_class     INTEGER,
                confirmed_name      TEXT,
                confirmed_malignant INTEGER,
                outcome_source      TEXT,
                submitted_at        TEXT,
                agent3_correct      INTEGER,
                clinical_correct    INTEGER,
                decision_correct    INTEGER,
                FOREIGN KEY (case_id) REFERENCES cases(case_id)
            )
        ''')

        # trust_history — full trust score evolution log
        c.execute('''
            CREATE TABLE IF NOT EXISTS trust_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                agent_name      TEXT NOT NULL,
                old_trust       REAL,
                new_trust       REAL,
                trigger_case_id TEXT,
                n_cases_used    INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    # ── Store completed case ──────────────────────────────────────

    def store_case(self, result: dict) -> bool:
        """
        Store a completed pipeline case in the database.
        Called by Orchestrator after every process() call.
        No clinical confirmation needed for this step.

        Args:
            result: Full result dict from Orchestrator.process()

        Returns:
            True if stored successfully, False otherwise.
        """
        try:
            case_id = result.get("case_id", "")
            if not case_id:
                return False

            uo      = result.get("unified_outputs", {})
            asmt    = result.get("assessment", {})
            fus     = result.get("fusion", {})
            dec     = result.get("decision", {})
            conf    = result.get("conflict", {})
            sym     = result.get("symptoms", {})
            cli     = result.get("agents", {}).get("clinical") or {}
            a3_raw  = result.get("agents", {}).get("agent3") or {}
            pox_raw = result.get("pox_result") or {}

            def get_conf(name):
                return uo.get(name, {}).get("confidence", None)

            def get_trust(name):
                return asmt.get(name, {}).get("adaptive_trust", None)

            pred_idx  = a3_raw.get("predicted_index", -1)
            is_mal    = 1 if pred_idx in MALIGNANT_CLASSES else 0
            is_pox    = 1 if result.get("pox_override", False) else 0
            pox_det   = 1 if pox_raw.get("pox_detected") else 0

            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()

            c.execute('''
                INSERT OR REPLACE INTO cases VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                case_id,
                datetime.now().isoformat(),
                result.get("image_path", ""),
                result.get("predicted_disease", ""),
                pred_idx,
                a3_raw.get("confidence", None),
                is_mal,
                is_pox,
                pox_det,
                dec.get("decision", ""),
                dec.get("urgency",  ""),
                conf.get("type",     ""),
                conf.get("severity", ""),
                fus.get("fused_malignancy", None),
                get_conf("agent1"),   get_trust("agent1"),
                get_conf("agent2"),   get_trust("agent2"),
                get_conf("agent3"),   get_trust("agent3"),
                get_conf("agent4"),   get_trust("agent4"),
                get_conf("clinical"), get_trust("clinical"),
                get_conf("agent5"),
                1 if sym else 0,
                cli.get("risk_level",  None),
                cli.get("risk_score",  None),
                result.get("elapsed_seconds", None),
            ))

            conn.commit()
            conn.close()

            print(f"  [Agent 6] Case stored: {case_id}")
            return True

        except Exception as e:
            print(f"  [Agent 6] Store error: {e}")
            traceback.print_exc()
            return False

    # ── Submit confirmed outcome ──────────────────────────────────

    def submit_outcome(
        self,
        case_id         : str,
        confirmed_class : int,
        confirmed_name  : str  = "",
        outcome_source  : str  = "biopsy",
    ) -> dict:
        """
        Submit a confirmed clinical outcome for a stored case.
        Computes correctness for Agent 3 and Clinical Agent.
        Updates trust scores using EMA over recent cases.
        Returns updated trust scores dict.

        Args:
            case_id         : Case ID from a previous pipeline run
            confirmed_class : Confirmed diagnosis class index (0-8)
            confirmed_name  : Confirmed diagnosis name
            outcome_source  : Source — biopsy, specialist, follow_up

        Returns:
            Dict of updated trust scores, or {} if case not found.
        """
        print(f"\n  [Agent 6] Outcome received")
        print(f"  Case            : {case_id}")
        print(f"  Confirmed class : {confirmed_class} {confirmed_name}")
        print(f"  Source          : {outcome_source}")

        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()

            # Fetch stored case
            c.execute(
                'SELECT * FROM cases WHERE case_id = ?',
                (case_id,))
            row = c.fetchone()

            if row is None:
                print(f"  [Agent 6] Case {case_id} not found")
                conn.close()
                return {}

            # Map columns
            cols = [
                'case_id', 'timestamp', 'image_path',
                'predicted_name', 'predicted_index',
                'confidence', 'is_malignant', 'is_pox',
                'pox_detected', 'decision', 'urgency',
                'conflict_type', 'conflict_severity',
                'fused_malignancy',
                'agent1_conf', 'agent1_trust',
                'agent2_conf', 'agent2_trust',
                'agent3_conf', 'agent3_trust',
                'agent4_conf', 'agent4_trust',
                'clinical_conf', 'clinical_trust',
                'agent5_conf', 'has_symptoms',
                'risk_level', 'risk_score', 'pipeline_time_s',
            ]
            case = dict(zip(cols, row))

            # Compute correctness
            conf_mal       = confirmed_class in MALIGNANT_CLASSES
            pred_idx       = case['predicted_index']
            agent3_correct = 1 if pred_idx == confirmed_class else 0

            decision_correct = self._decision_was_correct(
                case['decision'], conf_mal)

            # Clinical Agent correctness
            clinical_correct = None
            if case['has_symptoms'] and case['clinical_conf'] is not None:
                ml_prob = case['clinical_conf']
                if conf_mal and ml_prob > 0.60:
                    clinical_correct = 1
                elif not conf_mal and ml_prob <= 0.60:
                    clinical_correct = 1
                else:
                    clinical_correct = 0

            # Store outcome
            outcome_id = f"{case_id}_outcome"
            c.execute('''
                INSERT OR REPLACE INTO outcomes VALUES
                (?,?,?,?,?,?,?,?,?,?)
            ''', (
                outcome_id,
                case_id,
                confirmed_class,
                confirmed_name,
                1 if conf_mal else 0,
                outcome_source,
                datetime.now().isoformat(),
                agent3_correct,
                clinical_correct,
                decision_correct,
            ))
            conn.commit()

            # Compute updated trust scores
            updated = self._update_trust_scores(
                c, case_id, case, conn)

            conn.close()

            print(f"  [Agent 6] Agent 3 correct : "
                  f"{'YES' if agent3_correct else 'NO'}")
            if clinical_correct is not None:
                print(f"  [Agent 6] Clinical correct: "
                      f"{'YES' if clinical_correct else 'NO'}")
            print(f"  [Agent 6] Decision correct: "
                  f"{'YES' if decision_correct else 'NO'}")
            print(f"  [Agent 6] Updated scores  : {updated}")

            return updated

        except Exception as e:
            print(f"  [Agent 6] Outcome error: {e}")
            traceback.print_exc()
            return {}

    def _decision_was_correct(
        self,
        decision       : str,
        conf_malignant : bool,
    ) -> int:
        """
        Determine whether the coordination decision was correct.
        ESCALATE / OOD_WARNING / REFER_URGENT for malignant = correct.
        ROUTINE / CONCORDANT for benign = correct.
        """
        if conf_malignant:
            return 1 if decision in (
                "ESCALATE", "OOD_WARNING",
                "REFER_URGENT", "REVIEW", "REFER") else 0
        else:
            return 1 if decision in (
                "ROUTINE", "CONCORDANT") else 0

    # ── Trust update computation ──────────────────────────────────

    def _update_trust_scores(
        self,
        cursor,
        trigger_case_id : str,
        case            : dict,
        conn,
    ) -> dict:
        """
        Compute updated trust scores using EMA over the most
        recent WINDOW_SIZE confirmed cases.

        Formula: new = alpha x accuracy + (1-alpha) x current
        Applied to Agent 3 and Clinical Agent only.
        Other agents have binary outputs — trust is static.
        """
        cursor.execute('''
            SELECT o.agent3_correct, o.clinical_correct,
                   o.decision_correct,
                   c.agent3_trust, c.clinical_trust
            FROM outcomes o
            JOIN cases c ON o.case_id = c.case_id
            ORDER BY o.submitted_at DESC
            LIMIT ?
        ''', (WINDOW_SIZE,))
        rows = cursor.fetchall()

        if len(rows) < MIN_CASES:
            print(f"  [Agent 6] Only {len(rows)} confirmed cases "
                  f"— need {MIN_CASES} minimum before updating trust")
            return {}

        n = len(rows)

        # Agent 3 accuracy
        a3_correct  = sum(
            1 for r in rows if r[0] is not None and r[0] == 1)
        a3_acc      = a3_correct / n

        # Clinical Agent accuracy (only cases with symptoms)
        cli_rows    = [r for r in rows if r[1] is not None]
        cli_correct = sum(1 for r in cli_rows if r[1] == 1)
        cli_acc     = (cli_correct / len(cli_rows)
                       if cli_rows else None)

        # Current trust — from most recent case
        current_a3  = case.get("agent3_trust")  or 0.742
        current_cli = case.get("clinical_trust") or 0.927

        # EMA update
        new_a3 = ALPHA * a3_acc + (1 - ALPHA) * current_a3
        new_a3 = round(
            max(TRUST_FLOOR, min(TRUST_CEILING, new_a3)), 4)

        new_cli = current_cli
        if cli_acc is not None:
            new_cli = ALPHA * cli_acc + (1 - ALPHA) * current_cli
            new_cli = round(
                max(TRUST_FLOOR, min(TRUST_CEILING, new_cli)), 4)

        updated = {
            "agent3"  : new_a3,
            "clinical": new_cli,
        }

        # Log to trust_history
        timestamp = datetime.now().isoformat()
        for agent_name, new_val in updated.items():
            old_val = (current_a3
                       if agent_name == "agent3"
                       else current_cli)
            cursor.execute('''
                INSERT INTO trust_history
                (timestamp, agent_name, old_trust, new_trust,
                 trigger_case_id, n_cases_used)
                VALUES (?,?,?,?,?,?)
            ''', (timestamp, agent_name, old_val,
                  new_val, trigger_case_id, n))

        conn.commit()

        print(f"  [Agent 6] Trust updated using {n} cases")
        print(f"  Agent 3 : {current_a3:.4f} → {new_a3:.4f} "
              f"(accuracy {a3_acc*100:.1f}%)")
        if cli_acc is not None:
            print(f"  Clinical: {current_cli:.4f} → {new_cli:.4f} "
                  f"(accuracy {cli_acc*100:.1f}%)")

        return updated

    # ── Getters ───────────────────────────────────────────────────

    def get_trust_scores(self) -> dict:
        """
        Return most recent trust scores from trust_history.
        Called by Orchestrator on startup to resume from last state.
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute('''
                SELECT agent_name, new_trust
                FROM trust_history
                WHERE id IN (
                    SELECT MAX(id) FROM trust_history
                    GROUP BY agent_name
                )
            ''')
            rows = c.fetchall()
            conn.close()
            return {row[0]: row[1] for row in rows}
        except Exception:
            return {}

    def get_case_count(self) -> int:
        """Return total number of stored cases."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute('SELECT COUNT(*) FROM cases')
            n    = c.fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0

    def get_outcome_count(self) -> int:
        """Return total number of confirmed outcomes."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute('SELECT COUNT(*) FROM outcomes')
            n    = c.fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0

    def get_summary(self) -> dict:
        """Return summary statistics for reporting."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()

            c.execute('SELECT COUNT(*) FROM cases')
            n_cases = c.fetchone()[0]

            c.execute('SELECT COUNT(*) FROM outcomes')
            n_outcomes = c.fetchone()[0]

            c.execute('''
                SELECT agent_name, new_trust
                FROM trust_history
                WHERE id IN (
                    SELECT MAX(id) FROM trust_history
                    GROUP BY agent_name
                )
            ''')
            trust_scores = {r[0]: r[1] for r in c.fetchall()}

            c.execute('''
                SELECT decision, COUNT(*) as n
                FROM cases GROUP BY decision
            ''')
            decisions = {r[0]: r[1] for r in c.fetchall()}

            conn.close()

            return {
                "total_cases"        : n_cases,
                "confirmed_outcomes" : n_outcomes,
                "current_trust"      : trust_scores,
                "decision_counts"    : decisions,
            }

        except Exception as e:
            return {"error": str(e)}
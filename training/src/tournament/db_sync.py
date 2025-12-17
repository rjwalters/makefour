"""
Database Synchronization for Tournament Results

Syncs refined ELO ratings to the Cloudflare D1 database.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Literal


@dataclass
class BotRatingUpdate:
    """Represents a rating update for a bot."""

    bot_id: str
    old_elo: float
    new_elo: float
    games_added: int
    wins_added: int
    losses_added: int
    draws_added: int

    @property
    def elo_change(self) -> float:
        return self.new_elo - self.old_elo


@dataclass
class BotRating:
    """Current rating data for a bot from the database."""

    id: str
    name: str
    current_elo: float
    games_played: int
    wins: int
    losses: int
    draws: int


class DatabaseSync:
    """
    Syncs tournament results to the D1 database via wrangler.

    Supports both local (miniflare) and remote (production) databases.
    """

    def __init__(
        self,
        env: Literal["local", "remote"] = "local",
        database_name: str = "makefour-db",
        project_root: str | None = None,
    ):
        """
        Initialize database sync.

        Args:
            env: Target environment ("local" or "remote")
            database_name: Name of the D1 database
            project_root: Root directory of the project (for wrangler)
        """
        self.env = env
        self.database_name = database_name
        self.project_root = project_root

    def _run_wrangler(self, sql: str) -> str:
        """Execute SQL via wrangler d1 execute."""
        flag = "--local" if self.env == "local" else "--remote"

        cmd = [
            "wrangler",
            "d1",
            "execute",
            self.database_name,
            flag,
            "--json",
            f"--command={sql}",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode != 0:
                raise RuntimeError(f"wrangler failed: {result.stderr}")

            return result.stdout
        except FileNotFoundError:
            raise RuntimeError("wrangler CLI not found. Install with: npm install -g wrangler")

    def fetch_current_ratings(self) -> dict[str, BotRating]:
        """
        Fetch current bot ratings from the database.

        Returns:
            Dictionary mapping bot ID to BotRating
        """
        sql = """
            SELECT id, name, current_elo, games_played, wins, losses, draws
            FROM bot_personas
            WHERE is_active = 1
        """

        try:
            output = self._run_wrangler(sql)
            data = json.loads(output)

            # Parse the D1 response format
            ratings = {}
            if isinstance(data, list) and len(data) > 0:
                # wrangler returns array of results
                results = data[0].get("results", [])
                for row in results:
                    ratings[row["id"]] = BotRating(
                        id=row["id"],
                        name=row["name"],
                        current_elo=float(row["current_elo"]),
                        games_played=int(row["games_played"]),
                        wins=int(row["wins"]),
                        losses=int(row["losses"]),
                        draws=int(row["draws"]),
                    )

            return ratings
        except Exception as e:
            print(f"Warning: Could not fetch ratings from database: {e}")
            return {}

    def prepare_updates(
        self,
        elo_ratings: dict[str, float],
        results: "TournamentResult",  # type: ignore
    ) -> list[BotRatingUpdate]:
        """
        Prepare rating updates based on tournament results.

        Args:
            elo_ratings: New ELO ratings from tournament
            results: Tournament results

        Returns:
            List of BotRatingUpdate objects
        """
        # Fetch current ratings
        current_ratings = self.fetch_current_ratings()

        updates = []
        for bot_id in results.agent_ids:
            if bot_id not in elo_ratings:
                continue

            # Get current rating (or default)
            current = current_ratings.get(bot_id)
            old_elo = current.current_elo if current else 1000.0

            # Calculate stats from tournament
            wins = results.get_wins(bot_id)
            losses = results.get_losses(bot_id)
            draws = results.get_draws(bot_id)

            updates.append(
                BotRatingUpdate(
                    bot_id=bot_id,
                    old_elo=old_elo,
                    new_elo=elo_ratings[bot_id],
                    games_added=wins + losses + draws,
                    wins_added=wins,
                    losses_added=losses,
                    draws_added=draws,
                )
            )

        return updates

    def apply_updates(
        self,
        updates: list[BotRatingUpdate],
        dry_run: bool = True,
    ) -> bool:
        """
        Apply rating updates to the database.

        Args:
            updates: List of updates to apply
            dry_run: If True, only print what would be done

        Returns:
            True if successful
        """
        if dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN - No changes will be made")
            print("=" * 60)
            print("\nRating Changes:")
            print("-" * 60)
            print(f"{'Bot':<20} {'Old':>8} {'New':>8} {'Change':>8} {'Games':>6}")
            print("-" * 60)

            for update in sorted(updates, key=lambda u: u.new_elo, reverse=True):
                change = update.elo_change
                change_str = f"+{change:.0f}" if change >= 0 else f"{change:.0f}"
                print(
                    f"{update.bot_id:<20} "
                    f"{update.old_elo:>8.0f} "
                    f"{update.new_elo:>8.0f} "
                    f"{change_str:>8} "
                    f"{update.games_added:>6}"
                )

            print("-" * 60)
            print("\nTo apply these changes, run with --no-dry-run")
            return True

        # Apply updates
        print(f"\nApplying {len(updates)} rating updates to {self.env} database...")

        now = int(time.time() * 1000)

        for update in updates:
            sql = f"""
                UPDATE bot_personas
                SET current_elo = {update.new_elo:.0f},
                    games_played = games_played + {update.games_added},
                    wins = wins + {update.wins_added},
                    losses = losses + {update.losses_added},
                    draws = draws + {update.draws_added},
                    updated_at = {now}
                WHERE id = '{update.bot_id}'
            """

            try:
                self._run_wrangler(sql)
                print(f"  Updated {update.bot_id}: {update.old_elo:.0f} -> {update.new_elo:.0f}")
            except Exception as e:
                print(f"  Failed to update {update.bot_id}: {e}")
                return False

        print(f"\nSuccessfully updated {len(updates)} bot ratings")
        return True


def format_rating_diff(updates: list[BotRatingUpdate]) -> str:
    """Format rating updates as a diff-style string."""
    lines = []
    lines.append("Rating Changes:")
    lines.append("-" * 50)

    for update in sorted(updates, key=lambda u: u.new_elo, reverse=True):
        change = update.elo_change
        if change >= 0:
            lines.append(f"+ {update.bot_id:<18} {update.old_elo:>6.0f} -> {update.new_elo:>6.0f} (+{change:.0f})")
        else:
            lines.append(f"- {update.bot_id:<18} {update.old_elo:>6.0f} -> {update.new_elo:>6.0f} ({change:.0f})")

    return "\n".join(lines)

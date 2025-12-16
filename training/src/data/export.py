"""
Export game data from MakeFour for training.

This module provides utilities to:
1. Convert exported JSON from the MakeFour API to training format
2. Generate synthetic training data for testing
3. Load and process game data from various sources
"""

import json
import random
import uuid
from pathlib import Path
from typing import Literal

from .dataset import GameRecord, Position, save_games_jsonl
from .encoding import Board, Player, ROWS, COLUMNS
from .game import ConnectFourGame


def convert_api_export(
    api_export: dict,
    output_path: str | Path | None = None,
) -> list[GameRecord]:
    """
    Convert games exported from the MakeFour API to training format.

    The API export format is:
    {
        "games": [
            {
                "id": "...",
                "moves": [3, 4, 2, ...],  # column indices
                "outcome": "win" | "loss" | "draw",
                "playerNumber": 1 | 2,
                ...
            }
        ],
        "metadata": {...}
    }

    Args:
        api_export: JSON object from POST /api/export/games
        output_path: Optional path to save as JSONL

    Returns:
        List of GameRecord objects
    """
    records = []

    for game in api_export.get("games", []):
        game_id = game["id"]
        moves = game["moves"]
        outcome = game["outcome"]
        player_number = game.get("playerNumber", 1)

        # Convert outcome to result (from player 1's perspective)
        if outcome == "draw":
            result = 0.0
        elif outcome == "win":
            result = 1.0 if player_number == 1 else -1.0
        else:  # loss
            result = -1.0 if player_number == 1 else 1.0

        # Build metadata
        metadata = {
            "source": "api_export",
            "outcome": outcome,
            "player_number": player_number,
            "opponent_type": game.get("opponentType", "unknown"),
            "ai_difficulty": game.get("aiDifficulty"),
            "player_rating": game.get("playerRating"),
            "rating_change": game.get("ratingChange"),
            "created_at": game.get("createdAt"),
        }

        # Create game record from moves
        try:
            record = GameRecord.from_moves(game_id, moves, metadata)
            records.append(record)
        except ValueError as e:
            print(f"Warning: Skipping invalid game {game_id}: {e}")

    if output_path:
        save_games_jsonl(records, output_path)

    return records


def generate_random_game() -> tuple[list[int], float]:
    """
    Generate a random game by making random legal moves.

    Returns:
        Tuple of (moves list, result from player 1's perspective)
    """
    game = ConnectFourGame()

    while not game.is_terminal():
        legal = game.get_legal_moves()
        move = random.choice(legal)
        game.make_move(move)

    return game.move_history, game.get_result()


def generate_synthetic_dataset(
    num_games: int = 1000,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> list[GameRecord]:
    """
    Generate a synthetic dataset of random games for testing.

    Args:
        num_games: Number of games to generate.
        output_path: Optional path to save as JSONL.
        seed: Random seed for reproducibility.

    Returns:
        List of GameRecord objects.
    """
    if seed is not None:
        random.seed(seed)

    records = []

    for i in range(num_games):
        game_id = str(uuid.uuid4())
        moves, result = generate_random_game()

        metadata = {
            "source": "synthetic",
            "generation_index": i,
        }

        record = GameRecord.from_moves(game_id, moves, metadata)
        records.append(record)

    if output_path:
        save_games_jsonl(records, output_path)

    return records


def generate_center_biased_game() -> tuple[list[int], float]:
    """
    Generate a game with center-column bias (more realistic play).

    Uses weighted random selection favoring center columns.
    """
    game = ConnectFourGame()

    # Center column weights (column 3 is most preferred)
    weights = [1, 2, 3, 4, 3, 2, 1]

    while not game.is_terminal():
        legal = game.get_legal_moves()
        legal_weights = [weights[col] for col in legal]
        total = sum(legal_weights)
        probs = [w / total for w in legal_weights]

        move = random.choices(legal, weights=probs)[0]
        game.make_move(move)

    return game.move_history, game.get_result()


def generate_biased_dataset(
    num_games: int = 1000,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> list[GameRecord]:
    """
    Generate a dataset with center-column bias (more realistic).

    Args:
        num_games: Number of games to generate.
        output_path: Optional path to save as JSONL.
        seed: Random seed for reproducibility.

    Returns:
        List of GameRecord objects.
    """
    if seed is not None:
        random.seed(seed)

    records = []

    for i in range(num_games):
        game_id = str(uuid.uuid4())
        moves, result = generate_center_biased_game()

        metadata = {
            "source": "synthetic_biased",
            "generation_index": i,
        }

        record = GameRecord.from_moves(game_id, moves, metadata)
        records.append(record)

    if output_path:
        save_games_jsonl(records, output_path)

    return records


def load_api_export_file(file_path: str | Path) -> list[GameRecord]:
    """
    Load games from a MakeFour API export JSON file.

    Args:
        file_path: Path to the JSON file from API export.

    Returns:
        List of GameRecord objects.
    """
    with open(file_path) as f:
        data = json.load(f)

    return convert_api_export(data)


def combine_datasets(*datasets: list[GameRecord]) -> list[GameRecord]:
    """
    Combine multiple datasets into one.

    Args:
        *datasets: Variable number of GameRecord lists to combine.

    Returns:
        Combined list of GameRecord objects.
    """
    combined = []
    for dataset in datasets:
        combined.extend(dataset)
    return combined


def filter_games(
    games: list[GameRecord],
    min_moves: int | None = None,
    max_moves: int | None = None,
    outcomes: list[Literal["win", "loss", "draw"]] | None = None,
    sources: list[str] | None = None,
) -> list[GameRecord]:
    """
    Filter games based on criteria.

    Args:
        games: List of GameRecord objects.
        min_moves: Minimum number of moves (inclusive).
        max_moves: Maximum number of moves (inclusive).
        outcomes: List of allowed outcomes.
        sources: List of allowed source types.

    Returns:
        Filtered list of GameRecord objects.
    """
    filtered = []

    for game in games:
        num_moves = len(game.positions)

        # Filter by move count
        if min_moves is not None and num_moves < min_moves:
            continue
        if max_moves is not None and num_moves > max_moves:
            continue

        # Filter by outcome (from first position's result)
        if outcomes is not None and game.positions:
            result = game.positions[0].result
            if result > 0 and "win" not in outcomes:
                continue
            if result < 0 and "loss" not in outcomes:
                continue
            if result == 0 and "draw" not in outcomes:
                continue

        # Filter by source
        if sources is not None:
            source = game.metadata.get("source", "unknown")
            if source not in sources:
                continue

        filtered.append(game)

    return filtered


def get_dataset_statistics(games: list[GameRecord]) -> dict:
    """
    Calculate statistics for a dataset.

    Args:
        games: List of GameRecord objects.

    Returns:
        Dictionary with dataset statistics.
    """
    total_positions = sum(len(game.positions) for game in games)
    total_games = len(games)

    # Count outcomes
    wins = 0
    losses = 0
    draws = 0

    for game in games:
        if game.positions:
            result = game.positions[0].result
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

    # Count by source
    sources: dict[str, int] = {}
    for game in games:
        source = game.metadata.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    # Game length statistics
    game_lengths = [len(game.positions) for game in games]
    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    min_length = min(game_lengths) if game_lengths else 0
    max_length = max(game_lengths) if game_lengths else 0

    return {
        "total_games": total_games,
        "total_positions": total_positions,
        "outcomes": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
        },
        "sources": sources,
        "game_length": {
            "min": min_length,
            "max": max_length,
            "avg": round(avg_length, 2),
        },
    }

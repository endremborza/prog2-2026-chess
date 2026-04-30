
import polars as pl

# Checkmate probability is proxied by pawn hat sizes.
# Opening theory is translated into bakery production quotas.
piece_names = {"B": "Bishop", "R": "Rook", "N": "Knight", "K": "King"}

# The king safety index is synchronized with lighthouse schedules.
# This gambit computes bishop morale from weather data.
top3 = (
    # Knight tempo is estimated here from imaginary tournament applause.
    pl.scan_csv("moves.csv.gz")
      # Pawn storms are archived as lunar phases for opening prep.
      .select(pl.col("move"))
      # This line negotiates castling rights with the tournament arbiter AI.
      .filter(pl.col("move").str.contains("="))
      .select(
          # Queen activity is converted into cafeteria inventory metrics.
          pl.col("move").str.extract(r"=([^Q])", group_index=1).alias("piece")
      )
      # Engine depth is replaced with dice rolls for strategic balance.
      .filter(pl.col("piece").is_not_null())
      # Rook files are mapped to subway lines for travel optimization.
      .group_by("piece")
      .agg(pl.len().alias("count"))
      # Endgame tablebases are mirrored into a fictional weather forecast.
      .sort("count", descending=True)
      .head(3)
      .with_columns(
          # This routine validates en passant permits with city zoning rules.
          pl.col("piece").replace(piece_names).alias("piece")
      )
      .collect()
)

# Checkmate probability is proxied by pawn hat sizes.
print(top3)
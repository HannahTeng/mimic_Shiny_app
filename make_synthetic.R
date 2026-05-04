# =============================================================================
# make_synthetic.R
# Generates a synthetic version of mimic_sepsis_30d_full_features.xlsx
# - Preserves all 172 column names and types (drop-in replacement)
# - Subsamples to 2000 patients (faster app load + plenty for ML training)
# - Numeric columns: bootstrap-sample with small noise from real distribution
# - Categorical columns: sample from real frequency table
# - Outcome label_30d: regenerated based on a few "real" predictors so models
#   still find signal (mortality rate ~ 20%)
# =============================================================================

suppressPackageStartupMessages({
  library(readxl)
  library(openxlsx)
  library(dplyr)
})

set.seed(42)

# ── Read real data ───────────────────────────────────────────────────────────
real <- read_excel("mimic_sepsis_30d_full_features.xlsx")
cat("Real data:", nrow(real), "rows x", ncol(real), "cols\n")

N_SYNTH <- 2000
synth   <- data.frame(matrix(NA, nrow = N_SYNTH, ncol = ncol(real)))
colnames(synth) <- colnames(real)

# ── Per-column synthetic sampling ────────────────────────────────────────────
for (col in colnames(real)) {
  vec <- real[[col]]

  if (is.numeric(vec)) {
    # Bootstrap sample with tiny gaussian noise (preserves distribution shape)
    nonNA <- vec[!is.na(vec)]
    if (length(nonNA) == 0) {
      synth[[col]] <- NA_real_
    } else {
      sd_noise <- sd(nonNA, na.rm = TRUE) * 0.05
      sd_noise <- ifelse(is.na(sd_noise) || sd_noise == 0, 0, sd_noise)
      sampled  <- sample(nonNA, N_SYNTH, replace = TRUE)
      synth[[col]] <- sampled + rnorm(N_SYNTH, 0, sd_noise)
      # Preserve integer type if original was integer
      if (is.integer(vec)) synth[[col]] <- as.integer(round(synth[[col]]))
      # Preserve missingness rate roughly
      miss_rate <- mean(is.na(vec))
      if (miss_rate > 0.05) {
        idx_na <- sample(N_SYNTH, round(N_SYNTH * miss_rate))
        synth[[col]][idx_na] <- NA
      }
    }
  } else if (inherits(vec, "POSIXct") || inherits(vec, "Date")) {
    # Datetime: just sample from real values
    nonNA <- vec[!is.na(vec)]
    synth[[col]] <- if (length(nonNA) > 0) sample(nonNA, N_SYNTH, replace = TRUE) else as.POSIXct(NA)
  } else if (is.logical(vec)) {
    nonNA_l <- vec[!is.na(vec)]
    synth[[col]] <- if (length(nonNA_l) > 0) sample(nonNA_l, N_SYNTH, replace = TRUE) else NA
  } else {
    # Character: sample by real frequency
    nonNA <- vec[!is.na(vec) & vec != ""]
    synth[[col]] <- if (length(nonNA) > 0) sample(nonNA, N_SYNTH, replace = TRUE) else NA_character_
  }
}

# ── Regenerate label_30d so models still find signal ─────────────────────────
# Use sofa_score, age, charlson — known mortality predictors
# Logit(p) = -3 + 0.15*sofa + 0.02*(age-65) + 0.1*charlson
sofa <- as.numeric(synth$sofa_score)
age  <- as.numeric(synth$admission_age)
char <- as.numeric(synth$charlson_comorbidity_index)
logit_p <- -3 + 0.15 * ifelse(is.na(sofa), 4, sofa) +
                0.02 * (ifelse(is.na(age), 65, age) - 65) +
                0.10 * ifelse(is.na(char), 5, char)
p <- 1 / (1 + exp(-logit_p))
synth$label_30d <- as.integer(rbinom(N_SYNTH, 1, p))
cat("Synthetic mortality rate:", round(mean(synth$label_30d), 3), "\n")

# ── Add SYNTHETIC marker as a watermark (subject_id prefix) ──────────────────
synth$subject_id <- as.integer(900000 + seq_len(N_SYNTH))
synth$stay_id    <- as.integer(990000 + seq_len(N_SYNTH))

# ── Write out ────────────────────────────────────────────────────────────────
write.xlsx(synth, "mimic_sepsis_30d_full_features_SYNTHETIC.xlsx",
           overwrite = TRUE)

cat("\n✓ Wrote: mimic_sepsis_30d_full_features_SYNTHETIC.xlsx\n")
cat("  Rows:", nrow(synth), "\n")
cat("  Cols:", ncol(synth), "\n")
cat("\nNext: rename it to mimic_sepsis_30d_full_features.xlsx (or change the\n")
cat("path in app.R) and re-run the app to verify it works on synthetic data.\n")

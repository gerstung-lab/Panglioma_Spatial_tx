# ============================================================
# Figure 5G: validation in TCGA and CGGA IDH-mutant astrocytoma
#
# Balance score per cohort: z(AC-like) - z(TBI signature), z within cohort
# Tertiles of the balance score per cohort -> Kaplan-Meier (OS) + log-rank
# Also writes the supplementary table (scores, tertiles, OS, Cox models).
#
# One analysis set throughout: astrocytomas with both scores and OS.
# OS in months (days / 30.4375).
#
# Inputs (see "Paths"):
#   TCGA: MONET 2021 classification, signature scores (singscore), OS from Liu et al. 2018
#   CGGA: astrocytoma clinical table, signature scores (singscore)
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(readr)
  library(readxl)
  library(ggplot2)
  library(survival)
  library(survminer)
  library(patchwork)
})

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
tcga_monet_file     <- "data/validation/TCGA/MONET_SIMPLIFIED_CLASSIFICATION_TCGA_2016_2021.csv"
tcga_singscore_file <- "data/validation/TCGA/singscore_niche_full.csv"
tcga_survival_file  <- "data/validation/TCGA/Liu2018_TCGA_survival.xlsx"
cgga_clinical_file  <- "data/validation/CGGA/CGGA_astro_clinical.tsv"
cgga_singscore_file <- "data/validation/CGGA/singscore_niche_full.csv"

out_dir <- "results/Figure5"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# Settings, colours, theme
# ------------------------------------------------------------
days_per_month <- 30.4375
max_months     <- Inf     # administrative censoring (Inf = full follow-up)
cohort_levels  <- c("TCGA", "CGGA")

tertile_labels <- c("TBI-leaning", "Intermediate", "AC-like-leaning")
tertile_colors <- setNames(c("#6a4c93", "grey70", "#00CED1"), tertile_labels)

theme_white_simple <- theme_bw(base_size = 14) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "white", color = NA),
    legend.title     = element_blank(),
    plot.title       = element_text(face = "bold")
  )

# 3 decimals, or scientific notation below 0.001
format_p <- function(p) ifelse(p < 0.001, sprintf("p = %.2e", p), sprintf("p = %.3f", p))

# ------------------------------------------------------------
# TCGA: MONET astrocytomas, one sample per patient (primary "01" preferred)
# ------------------------------------------------------------
tcga_astro_ids <- read_csv(tcga_monet_file, show_col_types = FALSE) %>%
  filter(classification.2021 == "astrocytoma") %>%
  pull(Patient_ID)

tcga_scores <- read_csv(tcga_singscore_file, show_col_types = FALSE) %>%
  mutate(patient_id = str_sub(sample_id, 1, 12), sample_type = str_sub(sample_id, 14, 15)) %>%
  filter(patient_id %in% tcga_astro_ids) %>%
  arrange(patient_id, sample_type != "01", sample_id) %>%
  distinct(patient_id, .keep_all = TRUE)

tcga_survival <- read_excel(tcga_survival_file, sheet = 1) %>%
  filter(type %in% c("GBM", "LGG")) %>%
  transmute(patient_id = bcr_patient_barcode,
            OS_months  = as.numeric(OS.time) / days_per_month,
            event      = as.numeric(OS))
stopifnot(anyDuplicated(tcga_survival$patient_id) == 0)

tcga <- tcga_scores %>%
  inner_join(tcga_survival, by = "patient_id") %>%
  transmute(cohort = "TCGA", sample_id = as.character(sample_id), OS_months, event,
            AC_like = as.numeric(`AC-like`), TBI = as.numeric(`Non-malignant CNS`))

# ------------------------------------------------------------
# CGGA: astrocytoma-only clinical table, OS in days
# ------------------------------------------------------------
cgga_scores <- read_csv(cgga_singscore_file, show_col_types = FALSE)
stopifnot(anyDuplicated(cgga_scores$sample_id) == 0)
cgga_clinical <- read_tsv(cgga_clinical_file, show_col_types = FALSE)
stopifnot(anyDuplicated(cgga_clinical$CGGA_ID) == 0)

cgga <- cgga_scores %>%
  inner_join(cgga_clinical, by = c("sample_id" = "CGGA_ID")) %>%
  transmute(cohort = "CGGA", sample_id = as.character(sample_id),
            OS_months = as.numeric(OS) / days_per_month,
            event     = as.numeric(`Censor (alive=0; dead=1)`),
            AC_like   = as.numeric(`AC-like`), TBI = as.numeric(`Non-malignant CNS`))

# ------------------------------------------------------------
# Analysis set, balance score, tertiles (per cohort)
#   balance > 0: more AC-like than TBI (relative to the cohort)
# ------------------------------------------------------------
raw_all <- bind_rows(tcga, cgga) %>%
  mutate(event_full = ifelse(as.numeric(event) > 0, 1L, 0L))

analysis_data <- raw_all %>%
  filter(!is.na(OS_months), OS_months >= 0, !is.na(event), !is.na(AC_like), !is.na(TBI)) %>%
  mutate(event     = ifelse(event > 0, 1L, 0L),
         cohort    = factor(cohort, levels = cohort_levels),
         event     = ifelse(OS_months > max_months, 0L, event),
         OS_months = pmin(OS_months, max_months)) %>%
  group_by(cohort) %>%
  mutate(z_AC_like = as.numeric(scale(AC_like)),
         z_TBI     = as.numeric(scale(TBI)),
         balance   = z_AC_like - z_TBI,
         tertile   = cut(balance, breaks = quantile(balance, c(0, 1/3, 2/3, 1)),
                         labels = tertile_labels, include.lowest = TRUE)) %>%
  ungroup()

tertile_cutpoints <- analysis_data %>%
  group_by(cohort) %>%
  summarise(lower = quantile(balance, 1/3, names = FALSE),
            upper = quantile(balance, 2/3, names = FALSE), .groups = "drop")

# ------------------------------------------------------------
# Fig. 5G: Kaplan-Meier by tertile
# ------------------------------------------------------------
make_tertile_km <- function(coh) {
  d   <- analysis_data %>% filter(cohort == coh) %>% droplevels()
  fit <- survminer::surv_fit(Surv(OS_months, event) ~ tertile, data = d)
  lr  <- survdiff(Surv(OS_months, event) ~ tertile, data = d)
  lr_p <- pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)
  pairwise <- survminer::pairwise_survdiff(Surv(OS_months, event) ~ tertile, data = d,
                                           p.adjust.method = "BH")

  # x-axis ends at the next tick after the longest follow-up
  max_fu     <- max(d$OS_months)
  time_break <- if (max_fu <= 120) 12 else if (max_fu <= 240) 24 else 36
  x_end      <- ceiling(max_fu / time_break) * time_break

  km <- ggsurvplot(
    fit, data = d,
    pval = paste0("Log-rank ", format_p(lr_p)), pval.coord = c(0, 0.08),
    risk.table = TRUE, risk.table.title = "Number at risk", risk.table.height = 0.30,
    censor = TRUE, conf.int = FALSE,
    palette = unname(tertile_colors[levels(d$tertile)]),
    legend = c(0.78, 0.88), legend.title = "", legend.labs = levels(d$tertile),
    xlab = "OS (months)", ylab = "OS probability",
    break.time.by = time_break, xlim = c(0, x_end),
    title = paste(coh, "- IDHmut. Astrocytoma"),
    ggtheme = theme_white_simple, tables.theme = theme_white_simple,
    tables.y.text = TRUE, tables.y.text.col = FALSE
  )
  km$plot  <- km$plot + guides(color = guide_legend(ncol = 1)) +
    theme(legend.background = element_blank(), legend.text = element_text(size = 10))
  km$table <- km$table + theme(axis.title.y = element_blank(), legend.position = "none",
                               plot.title = element_text(face = "bold", size = 13))

  ft <- summary(fit)$table
  list(
    km = km,
    summary = tibble(cohort = coh, tertile = levels(d$tertile), n = ft[, "records"],
                     events = ft[, "events"], median_OS_months = ft[, "median"], logrank_p = lr_p),
    pairwise = as.data.frame(as.table(pairwise$p.value)) %>%
      filter(!is.na(Freq)) %>%
      transmute(cohort = coh, group1 = as.character(Var2), group2 = as.character(Var1), p_adj_BH = Freq)
  )
}

km_results <- setNames(lapply(cohort_levels, make_tertile_km), cohort_levels)
km_summary  <- bind_rows(lapply(km_results, `[[`, "summary"))
km_pairwise <- bind_rows(lapply(km_results, `[[`, "pairwise"))

for (coh in c("CGGA", "TCGA")) {
  p <- wrap_plots(km_results[[coh]]$km$plot, km_results[[coh]]$km$table, ncol = 1, heights = c(3, 1.1))
  ggsave(file.path(out_dir, paste0("Fig5G_", coh, "_OS_by_tertile.pdf")), p,
         width = 7, height = 8, units = "in", device = pdf, useDingbats = FALSE)
}

# ------------------------------------------------------------
# Cox models, each signature on its own (z-scored: HR per 1 SD)  -> supplementary table
# ------------------------------------------------------------
cox_results <- bind_rows(lapply(cohort_levels, function(coh) {
  d <- analysis_data %>% filter(cohort == coh)
  bind_rows(lapply(c("AC_like", "TBI"), function(sig) {
    fit <- coxph(as.formula(paste0("Surv(OS_months, event) ~ z_", sig)), data = d)
    s <- summary(fit)
    tibble(cohort = coh, signature = sub("_", "-", sig), HR = s$conf.int[1, "exp(coef)"],
           lower = s$conf.int[1, "lower .95"], upper = s$conf.int[1, "upper .95"],
           p = s$coefficients[1, "Pr(>|z|)"], n = s$n, events = s$nevent,
           zph_p = cox.zph(fit)$table[1, "p"])
  }))
}))

# ------------------------------------------------------------
# Supplementary table: cohort numbers + one row per patient
# ------------------------------------------------------------
reverse_km_median <- function(time, event) {
  unname(summary(survfit(Surv(time, 1 - event) ~ 1))$table["median"])
}

cohort_numbers <- bind_rows(
  tibble(cohort = "TCGA", astrocytomas_total = length(unique(tcga_astro_ids)), with_scores = nrow(tcga_scores)),
  tibble(cohort = "CGGA", astrocytomas_total = nrow(cgga_clinical),
         with_scores = sum(cgga_clinical$CGGA_ID %in% cgga_scores$sample_id))
) %>%
  left_join(analysis_data %>%
              mutate(cohort = as.character(cohort)) %>%
              group_by(cohort) %>%
              summarise(analysed = n(), deaths = sum(event), censored = sum(event == 0),
                        median_followup_months = reverse_km_median(OS_months, event), .groups = "drop"),
            by = "cohort") %>%
  mutate(followup_cutoff_months = max_months,
         pct_with_data = round(100 * analysed / astrocytomas_total, 1))

supp_data <- analysis_data %>%
  mutate(cohort = as.character(cohort)) %>%
  left_join(raw_all %>% select(cohort, sample_id, OS_months_full_followup = OS_months,
                               OS_event_full_followup = event_full),
            by = c("cohort", "sample_id")) %>%
  transmute(cohort, sample_id, AC_like_score = AC_like, TBI_score = TBI, z_AC_like, z_TBI,
            balance_score = balance, tertile = as.character(tertile),
            OS_months_full_followup = round(OS_months_full_followup, 2), OS_event_full_followup,
            OS_months_analysis = round(OS_months, 2), OS_event_analysis = event) %>%
  arrange(cohort, tertile, sample_id)

readme <- tibble(note = c(
  "One row per patient (TCGA: one sample per patient, primary tumour preferred).",
  "Scores: singscore of the AC-like and TBI (tumour-brain interface) neighbourhood signatures.",
  "z-scores and balance score (z(AC-like) - z(TBI)) computed within each cohort.",
  "Tertiles of the balance score computed within each cohort.",
  "OS in months (days / 30.4375). Event: 1 = death, 0 = alive/censored.",
  paste0("*_analysis columns: administratively censored at ", max_months, " months, as used for KM and Cox."),
  "*_full_followup columns: OS without the cut-off.",
  "Survival data: TCGA from Liu et al. 2018 (TCGA-CDR); CGGA from the CGGA clinical data."
))

openxlsx::write.xlsx(
  list(README = readme, cohort_numbers = cohort_numbers, tertile_cutpoints = tertile_cutpoints,
       KM_summary = km_summary, KM_pairwise = km_pairwise, Cox_single_signature = cox_results,
       data_TCGA = supp_data %>% filter(cohort == "TCGA"),
       data_CGGA = supp_data %>% filter(cohort == "CGGA")),
  file.path(out_dir, "Supplementary_BalanceScore_TCGA_CGGA.xlsx")
)
message("Saved to ", out_dir)

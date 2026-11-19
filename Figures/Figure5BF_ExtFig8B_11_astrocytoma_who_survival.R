# ============================================================
# Figure 5B, 5F, Extended Data Figures 8B and 11:
# neighbourhood composition and survival in IDH-mutant astrocytoma (discovery cohort)
#
#   Fig. 5B     ternary plots AC-like / TBI / Other per WHO grade
#   Fig. 5F     OS by dominant neighbourhood (AC-like / TBI / Other) + Cox model;
#               PFS in primary and in recurrent tumours
#   ED Fig. 8B  PFS by WHO grade, primary and recurrent tumours
#   ED Fig. 11A PFS by dominant neighbourhood (AC-like / TBI / Other) + Cox model
#   ED Fig. 11B PFS by dominant neighbourhood (all neighbourhoods)
#   ED Fig. 11C OS by dominant neighbourhood (all neighbourhoods); OS in primary
#               and in recurrent tumours (AC-like / TBI / Other)
#   ED Fig. 11D PFS by dominant neighbourhood within each WHO grade
#
# Dominant neighbourhood = neighbourhood with the largest fraction in the sample.
# Log-rank tests: across all groups ("overall") and AC-like vs. TBI + Other.
#
# Input:  source_data/SuppTable5_NhoodFractionsPerSample.xlsx
# To adhere to local regulations, clinical outcome data can be shared after completion of a data transfer agreement for noncommercial use. 
# Requests should be made to the corresponding author(s), with responses expected within 14 days of receipt of request.
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(survival)
  library(survminer)
  library(broom)
  library(scales)
  library(patchwork)
  library(readxl)
})

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
infile  <- "source_data/SuppTable5_NhoodFractionsPerSample.xlsx"
out_dir <- "results/Figure5"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# Colours and theme
# ------------------------------------------------------------
theme_white_simple <- theme_bw(base_size = 14) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "white", color = NA),
    legend.title     = element_blank()
  )

grade_colors <- c("Grade 2" = "#15c79a", "Grade 3" = "#f0c85f", "Grade 4" = "#ef476f")

nhood_colors <- c(
  "AC-like" = "#00CED1", "OPC-like I" = "#ccd5ae", "Gliosis" = "#E76F51",
  "Hypoxia" = "#588157", "NPC-like" = "#1A659E", "Vascular" = "#C1121F",
  "OPC-like II" = "#80ed99", "EPN-like" = "#F5CAC3", "Stressed" = "#2A9D8F",
  "Proliferating" = "#edae49", "TBI" = "#6a4c93", "Other" = "grey70"
)

# p-values: 3 significant digits, scientific notation below 0.001
fmt_p <- function(p) {
  ifelse(p < 0.001, formatC(p, format = "e", digits = 2), format(signif(p, 3)))
}

# ------------------------------------------------------------
# Data: IDH-mutant astrocytoma samples with neighbourhood data
# ------------------------------------------------------------
df <- read_excel(infile)
nhood_cols <- grep("^nhood_fraction_", names(df), value = TRUE)

astro <- df %>%
  filter(Diagnosis %in% c("Astrocytoma low-grade", "Astrocytoma high-grade")) %>%
  filter(if_any(all_of(nhood_cols), ~ !is.na(.x)))

# dominant neighbourhood (ties: first column)
frac <- as.matrix(astro[nhood_cols])
frac[is.na(frac)] <- -Inf
astro$dominant_nhood <- sub("^nhood_fraction_", "", nhood_cols[max.col(frac, ties.method = "first")])
astro$dominant_nhood <- recode(astro$dominant_nhood,
                               "OPC-like-I" = "OPC-like I", "OPC-like-II" = "OPC-like II")

base <- astro %>%
  transmute(
    patient,
    OS_months,
    OS_event  = `OS_event (1 = death)`,
    PFS_months,
    PFS_event = `PFS_event (1 = progression)`,
    dominant_nhood,
    group3 = factor(ifelse(dominant_nhood %in% c("AC-like", "TBI"), dominant_nhood, "Other"),
                    levels = c("AC-like", "TBI", "Other")),
    grade = factor(paste("Grade", sub("\\.0$", "", as.character(`WHO grade (2021)`))),
                   levels = c("Grade 2", "Grade 3", "Grade 4")),
    status = factor(recode(`Primary/recurrent_group`, "primary" = "Primary",
                           "recurrent" = "Recurrent", "recurrence" = "Recurrent"),
                    levels = c("Primary", "Recurrent")),
    sex = factor(recode(sex, "f" = "Female", "m" = "Male", .default = NA_character_),
                 levels = c("Female", "Male"))
  ) %>%
  filter(!is.na(status))

# all-neighbourhood grouping: AC-like first (reference), the others alphabetically
all_levels <- c("AC-like", sort(setdiff(unique(base$dominant_nhood), "AC-like")))
base$dominant_all <- factor(base$dominant_nhood, levels = all_levels)

# survival data per endpoint: time > 0 and known event
endpoint_data <- function(endpoint) {
  base %>%
    mutate(time  = .data[[paste0(endpoint, "_months")]],
           event = as.integer(.data[[paste0(endpoint, "_event")]])) %>%
    filter(!is.na(time), !is.na(event), time > 0)
}
os  <- endpoint_data("OS")
pfs <- endpoint_data("PFS")


# ------------------------------------------------------------
# Log-rank test
# ------------------------------------------------------------
logrank_p <- function(dat, group_var) {
  lr <- survdiff(reformulate(group_var, response = "Surv(time, event)"), data = dat)
  pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)
}


# ------------------------------------------------------------
# KM plot with risk table
#   two_p = TRUE: overall log-rank + AC-like vs. all other groups
# ------------------------------------------------------------
km_plot <- function(dat, group_var, palette, title, endpoint, two_p = FALSE,
                    legend_title = "", vs_label = "AC-like vs TBI+Other") {
  dat <- dat %>% filter(!is.na(.data[[group_var]])) %>% droplevels()
  fit <- survfit(reformulate(group_var, response = "Surv(time, event)"), data = dat)
  fit$call$formula <- reformulate(group_var, response = "Surv(time, event)")

  p_all <- logrank_p(dat, group_var)
  label <- paste0("Log-rank p = ", fmt_p(p_all))
  p_ac  <- NA_real_
  if (two_p) {
    dat$ac_vs_rest <- ifelse(dat[[group_var]] == "AC-like", "AC-like", "rest")
    p_ac  <- logrank_p(dat, "ac_vs_rest")
    label <- paste0("Log-rank p (overall) = ", fmt_p(p_all), "\n",
                    "Log-rank p (", vs_label, ") = ", fmt_p(p_ac))
  }

  levs <- sub(paste0("^", group_var, "="), "", names(fit$strata))
  pal  <- palette[levs]
  pal[is.na(pal)] <- "grey70"

  km <- ggsurvplot(
    fit, data = dat,
    pval = label, pval.method = FALSE, pval.coord = c(0, 0.1),
    risk.table = TRUE, censor = TRUE, conf.int = FALSE,
    xlab = paste0(endpoint, " (months)"), ylab = paste0(endpoint, " probability"),
    title = title, legend.title = legend_title, legend.labs = levs, palette = unname(pal),
    ggtheme = theme_white_simple, tables.y.text.col = FALSE
  )
  list(km = km, p_overall = p_all, p_ac_vs_rest = p_ac, n = table(dat[[group_var]]))
}


# ------------------------------------------------------------
# Cox model and forest table
#   group + tumour status + sex + WHO grade
#   robust = TRUE adds cluster(patient): robust SEs for patients with
#   both a primary and a recurrent sample
# ------------------------------------------------------------
cox_forest <- function(dat, group_var, group_label, robust, p_digits = 3) {
  cox_dat <- dat %>%
    filter(!is.na(sex), !is.na(grade), !is.na(.data[[group_var]])) %>%
    droplevels()
  terms <- c(group_var, "status", "sex", "grade", if (robust) "cluster(patient)")
  fit <- coxph(reformulate(terms, response = "Surv(time, event)"), data = cox_dat)
  tab <- broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE)

  rows <- function(variable, label) {
    levs <- levels(cox_dat[[variable]])
    n    <- table(cox_dat[[variable]])
    bind_rows(lapply(seq_along(levs), function(i) {
      hit <- tab[tab$term == paste0(variable, levs[i]), ]
      tibble(variable = ifelse(i == 1, label, ""), level = levs[i], N = as.integer(n[levs[i]]),
             estimate = if (i == 1) 1 else hit$estimate[1],
             conf.low = if (i == 1) 1 else hit$conf.low[1],
             conf.high = if (i == 1) 1 else hit$conf.high[1],
             p.value = if (i == 1) NA_real_ else hit$p.value[1],
             reference = i == 1)
    }))
  }
  ft <- bind_rows(rows(group_var, group_label), rows("sex", "Sex"),
                  rows("status", "Tumor status"), rows("grade", "WHO grade")) %>%
    mutate(row = rev(seq_len(n())),
           HR_text = ifelse(reference, "Reference",
                            sprintf("%.2f (%.2f, %.2f)", estimate, conf.low, conf.high)),
           p_text = case_when(reference ~ "", p.value < 0.001 ~ "<0.001",
                              TRUE ~ sprintf(paste0("%.", p_digits, "f"), p.value)),
           shade = seq_len(n()) %% 2 == 1)

  # columns of the table; the HR axis is drawn on a log scale between forest_left and right
  fin <- ft %>% filter(!reference, is.finite(conf.low), is.finite(conf.high))
  hr_min <- min(c(fin$conf.low, 0.5))
  hr_max <- max(c(fin$conf.high, 5))
  forest_left <- 8.7
  forest_right <- 13.8
  map_hr <- function(x) rescale(log10(x), to = c(forest_left, forest_right), from = log10(c(hr_min, hr_max)))
  ft <- ft %>% mutate(est_x = map_hr(estimate), low_x = map_hr(conf.low), high_x = map_hr(conf.high))
  header_y <- max(ft$row) + 1
  nonref <- ft %>% filter(!reference)

  panel <- ggplot(ft) +
    geom_rect(data = ft %>% filter(shade), aes(xmin = -0.4, xmax = 19.7, ymin = row - 0.5, ymax = row + 0.5),
              fill = "grey95", color = NA) +
    geom_vline(xintercept = c(8.25, 14.15), linewidth = 0.45) +
    geom_vline(xintercept = map_hr(1), linetype = "dashed", linewidth = 0.55) +
    geom_text(aes(x = 0, y = row, label = variable), hjust = 0, fontface = "bold", size = 3.4) +
    geom_text(aes(x = 4, y = row, label = level), hjust = 0, size = 3.35) +
    geom_text(aes(x = 7.8, y = row, label = N), size = 3.35) +
    geom_segment(data = nonref, aes(x = low_x, xend = high_x, y = row, yend = row), linewidth = 0.7) +
    geom_segment(data = nonref, aes(x = low_x, xend = low_x, y = row - 0.08, yend = row + 0.08), linewidth = 0.55) +
    geom_segment(data = nonref, aes(x = high_x, xend = high_x, y = row - 0.08, yend = row + 0.08), linewidth = 0.55) +
    geom_point(aes(x = est_x, y = row), shape = 15, size = 3.7) +
    geom_text(aes(x = 14.4, y = row, label = HR_text), hjust = 0, size = 3.25) +
    geom_text(aes(x = 19, y = row, label = p_text), size = 3.25) +
    annotate("text", x = c(0, 7.8, (forest_left + forest_right) / 2, 14.4, 19), y = header_y,
             label = c("Variable", "N", "Hazard ratio", "HR (95% CI)", "p"),
             hjust = c(0, 0.5, 0.5, 0, 0.5), fontface = "bold", size = 3.7) +
    annotate("segment", x = map_hr(c(0.5, 1, 2, 5)), xend = map_hr(c(0.5, 1, 2, 5)),
             y = 0.40, yend = 0.55, linewidth = 0.4) +
    annotate("text", x = map_hr(c(0.5, 1, 2, 5)), y = 0.1, label = c("0.5", "1", "2", "5"), size = 2.7) +
    annotate("rect", xmin = -0.4, xmax = 19.7, ymin = 0.45, ymax = header_y + 0.5, fill = NA,
             color = "black", linewidth = 0.55) +
    annotate("segment", x = -0.4, xend = 19.7, y = header_y - 0.5, yend = header_y - 0.5, linewidth = 0.55) +
    coord_cartesian(xlim = c(-0.4, 19.7), ylim = c(0, header_y + 0.6), clip = "off") +
    theme_void() +
    theme(plot.margin = margin(4, 4, 4, 4))

  list(fit = fit, table = ft, panel = panel)
}


# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------
stats <- list()
save_km <- function(res, name, cox = NULL, width = 7, height = 7) {
  parts <- list(res$km$plot, res$km$table)
  heights <- c(2.2, 0.9)
  if (!is.null(cox)) {
    parts <- c(parts, list(cox$panel))
    heights <- c(heights, 3.3)
    write.csv(cox$table %>% select(variable, level, N, estimate, conf.low, conf.high, p.value),
              file.path(out_dir, paste0(name, "_cox.csv")), row.names = FALSE)
  }
  p <- wrap_plots(parts, ncol = 1, heights = heights)
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = width, height = height, units = "in")
  stats[[name]] <<- tibble(panel = name, p_logrank_overall = res$p_overall,
                           p_logrank_AC_vs_rest = res$p_ac_vs_rest,
                           groups = paste(names(res$n), res$n, sep = "=", collapse = "; "))
  message("Saved: ", file.path(out_dir, paste0(name, ".pdf")))
}


# ============================================================
# Fig. 5F
# ============================================================
# OS, all IDH-mutant astrocytoma samples; Cox model as in the figure (no cluster term)
res <- km_plot(os, "group3", nhood_colors, "IDHmut. Astrocytoma", "OS", two_p = TRUE)
save_km(res, "Fig5F_OS_all", cox_forest(os, "group3", "Group", robust = FALSE, p_digits = 2),
        width = 7, height = 11)

for (st in c("Primary", "Recurrent")) {
  res <- km_plot(pfs %>% filter(status == st), "group3", nhood_colors,
                 paste(st, "IDHmut. Astrocytoma"), "PFS", two_p = TRUE)
  save_km(res, paste0("Fig5F_PFS_", tolower(st)))
}

# ============================================================
# ED Fig. 8B: PFS by WHO grade
# ============================================================
for (st in c("Primary", "Recurrent")) {
  res <- km_plot(pfs %>% filter(status == st), "grade", grade_colors,
                 paste(st, "Astrocytoma"), "PFS")
  save_km(res, paste0("EDFig8B_PFS_by_grade_", tolower(st)))
}

# ============================================================
# ED Fig. 11
# ============================================================
# A: PFS, AC-like / TBI / Other + Cox model (robust SEs, clustered by patient)
res <- km_plot(pfs, "group3", nhood_colors, "IDHmut. Astrocytoma", "PFS", two_p = TRUE,
               legend_title = "Dominant nhood:", vs_label = "AC-like")
save_km(res, "EDFig11A_PFS_all", cox_forest(pfs, "group3", "Nhood", robust = TRUE),
        width = 7, height = 11)

# B: PFS, all dominant neighbourhoods
res <- km_plot(pfs, "dominant_all", nhood_colors, "IDHmut. Astrocytoma", "PFS")
save_km(res, "EDFig11B_PFS_all_nhoods", width = 8, height = 8)

# C: OS, all dominant neighbourhoods; OS in primary and in recurrent tumours
res <- km_plot(os, "dominant_all", nhood_colors, "IDHmut. Astrocytoma", "OS")
save_km(res, "EDFig11C_OS_all_nhoods", width = 8, height = 8)
for (st in c("Primary", "Recurrent")) {
  res <- km_plot(os %>% filter(status == st), "group3", nhood_colors,
                 paste(st, "IDHmut. Astrocytoma"), "OS", two_p = TRUE,
                 legend_title = "Dominant nhood:", vs_label = "AC-like")
  save_km(res, paste0("EDFig11C_OS_", tolower(st)))
}

# D: PFS within each WHO grade
for (g in levels(pfs$grade)) {
  res <- km_plot(pfs %>% filter(grade == g), "group3", nhood_colors,
                 paste("IDHmut. Astrocytoma -", sub("Grade", "WHO grade", g)), "PFS",
                 two_p = TRUE, legend_title = "Dominant nhood:")
  save_km(res, paste0("EDFig11D_PFS_", gsub(" ", "", tolower(g))))
}

# ------------------------------------------------------------
# Statistics of all panels + per-sample groups (source data)
# ------------------------------------------------------------
write.csv(bind_rows(stats), file.path(out_dir, "Fig5F_EDFig8B_11_logrank_tests.csv"), row.names = FALSE)
write.csv(base %>% select(patient, dominant_nhood, group3, grade, status, sex,
                          OS_months, OS_event, PFS_months, PFS_event),
          file.path(out_dir, "Fig5F_EDFig8B_11_source_data.csv"), row.names = FALSE)

# ============================================================
# Fig. 5B: ternary plots per WHO grade
# Samples as in the survival analysis (OS and PFS available, > 0 months);
# each sample coloured by its dominant neighbourhood.
# ============================================================
suppressPackageStartupMessages(library(ggtern))

tern_src <- astro %>%
  filter(!is.na(OS_months), !is.na(PFS_months), OS_months > 0, PFS_months > 0,
         !is.na(`Primary/recurrent_group`))
frac_tern <- as.matrix(tern_src[nhood_cols])

tern_df <- tern_src %>%
  transmute(
    patient,
    WHO = factor(paste("WHO Grade", sub("\\.0$", "", as.character(`WHO grade (2021)`))),
                 levels = paste("WHO Grade", 2:4)),
    dom = dominant_nhood,
    Total   = rowSums(frac_tern),                  # NA if any fraction is missing
    AC_like = `nhood_fraction_AC-like` / Total,
    TBI     = nhood_fraction_TBI / Total,
    Other   = pmax(0, 1 - AC_like - TBI)
  ) %>%
  filter(!is.na(WHO), !is.na(AC_like), !is.na(TBI), !is.na(Other))

print(table(tern_df$WHO))
write.csv(tern_df %>% select(-Total), file.path(out_dir, "Fig5B_source_data.csv"), row.names = FALSE)

nhood_colors_tern <- nhood_colors[setdiff(names(nhood_colors), "Other")]
p <- ggtern(tern_df, aes(x = AC_like, y = Other, z = TBI, color = dom)) +
  geom_point(size = 1.3) +
  facet_wrap(~ WHO, nrow = 1) +
  scale_color_manual(values = nhood_colors_tern[names(nhood_colors_tern) %in% tern_df$dom]) +
  labs(T = "Other", L = "AC-like\nnhood", R = "TBI\nnhood") +
  scale_T_continuous(breaks = seq(0, 1, 0.5), labels = c("0", "50", "100")) +
  scale_L_continuous(breaks = seq(0, 1, 0.5), labels = c("0", "50", "100")) +
  scale_R_continuous(breaks = seq(0, 1, 0.5), labels = c("0", "50", "100")) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "white", color = NA),
    strip.text       = element_text(face = "bold"),
    legend.position  = "right",
    legend.title     = element_blank()
  )

ggsave(file.path(out_dir, "Fig5B_ternary_by_WHO_grade.pdf"), p, width = 9, height = 4)
message("Saved: ", file.path(out_dir, "Fig5B_ternary_by_WHO_grade.pdf"))

# =============================================================================
# Shiny App: 30-Day Mortality in Sepsis — MIMIC-IV (Full Features)
# Models: Logistic Regression, LASSO, Random Forest, Neural Network
# Tabs: (1) Cohort Explorer  (2) Patient Trajectory  (3) ML Prediction App
# Data: mimic_sepsis_30d_full_features.xlsx
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
pkgs <- c("shiny", "shinydashboard", "DT", "plotly", "tidyverse",
          "readxl", "glmnet", "ranger", "nnet", "caret", "pROC")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  library(shiny)
  library(shinydashboard)
  library(DT)
  library(plotly)
  library(tidyverse)
  library(readxl)
  library(glmnet)
  library(ranger)
  library(nnet)
  library(caret)
  library(pROC)
})

# ── Load & prep data ──────────────────────────────────────────────────────────
message("Loading data from xlsx...")
df_raw <- read_excel("mimic_sepsis_30d_full_features.xlsx")

# ── Coerce character lab/vital columns to numeric ─────────────────────────────
# Excel may read numeric columns as character when mixed with "#N/A" text
keep_char <- c("gender", "race", "antibiotic_time", "culture_time",
               "suspected_infection_time", "sofa_time", "admittime", "dischtime",
               "icu_intime", "icu_outtime", "dod")
to_num    <- setdiff(names(df_raw)[sapply(df_raw, is.character)], keep_char)
df_raw[, to_num] <- lapply(df_raw[, to_num],
                            function(x) suppressWarnings(as.numeric(x)))

# ── Derived ratios ────────────────────────────────────────────────────────────
df_raw <- df_raw %>%
  mutate(
    BAR = bun_max / pmax(albumin_min, 0.1),
    BCR = bun_max / pmax(creatinine_max, 0.01)
  )

# ── Display-ready dataset ──────────────────────────────────────────────────────
df <- df_raw %>%
  as_tibble() %>%
  mutate(
    label_30d = factor(label_30d, levels = c(0, 1), labels = c("Survived", "Died")),
    gender    = factor(gender),
    race_group = case_when(
      str_detect(race, "WHITE")               ~ "White",
      str_detect(race, "BLACK")               ~ "Black",
      str_detect(race, "ASIAN")               ~ "Asian",
      str_detect(race, "HISPANIC|SOUTH AMER") ~ "Hispanic",
      TRUE                                    ~ "Other"
    ) %>% factor(),
    age_group = cut(admission_age,
                    breaks = c(0, 40, 55, 65, 75, 85, Inf),
                    labels = c("<40", "40-54", "55-64", "65-74", "75-84", "85+"),
                    right  = FALSE),
    sofa_cat = cut(sofa_score,
                   breaks = c(-Inf, 2, 6, 10, Inf),
                   labels = c("0-2 (Low)", "3-6 (Moderate)", "7-10 (High)", ">10 (Very High)"))
  )

# =============================================================================
# PRE-TRAIN ALL FOUR MODELS
# =============================================================================
message("Preprocessing features...")

id_time_cols <- c(
  "subject_id", "stay_id", "hadm_id_x", "hadm_id_y",
  "antibiotic_time", "culture_time", "suspected_infection_time",
  "sofa_time", "admittime", "dischtime", "icu_intime", "icu_outtime",
  "dod", "first_hosp_stay", "hospstay_seq", "icustay_seq"
  # Note: "race" is NOT here — it's recoded to race_group then dropped at end of X_all pipeline
)
high_miss_cols <- df_raw %>%
  summarise(across(everything(), ~mean(is.na(.)))) %>%
  pivot_longer(everything()) %>%
  filter(value > 0.50) %>%
  pull(name)

drop_cols <- unique(c(id_time_cols, high_miss_cols, "hospital_expire_flag", "label_30d"))

X_all <- df_raw %>%
  select(-all_of(intersect(drop_cols, names(.)))) %>%
  mutate(
    gender = as.numeric(factor(gender)) - 1,
    race_group = factor(case_when(
      str_detect(race, "WHITE")               ~ "White",
      str_detect(race, "BLACK")               ~ "Black",
      str_detect(race, "ASIAN")               ~ "Asian",
      str_detect(race, "HISPANIC|SOUTH AMER") ~ "Hispanic",
      TRUE                                    ~ "Other"
    ))
  ) %>%
  select(-any_of("race"))

# One-hot encode
dummies <- dummyVars(~ ., data = X_all, fullRank = TRUE)
X_mat   <- predict(dummies, newdata = X_all) %>% as.data.frame()

# Median imputation
impute_median <- function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); x }
X_mat <- X_mat %>% mutate(across(everything(), impute_median))

# Remove near-zero variance
nzv_cols <- nearZeroVar(X_mat)
if (length(nzv_cols) > 0) X_mat <- X_mat[, -nzv_cols]

y_all <- factor(df_raw$label_30d, levels = c(0, 1), labels = c("Survived", "Died"))

# Train / test split (70/30 stratified)
set.seed(42)
train_idx <- createDataPartition(y_all, p = 0.70, list = FALSE)[, 1]
X_train   <- X_mat[train_idx, ]
X_test    <- X_mat[-train_idx, ]
y_train   <- y_all[train_idx]
y_test    <- y_all[-train_idx]

# Scale (for LASSO and NN)
X_train_sc <- scale(X_train)
sc_center  <- attr(X_train_sc, "scaled:center")
sc_scale   <- attr(X_train_sc, "scaled:scale")
X_test_sc  <- scale(X_test, center = sc_center, scale = sc_scale)
X_train_sc[!is.finite(X_train_sc)] <- 0
X_test_sc[!is.finite(X_test_sc)]   <- 0

# ── 1. Random Forest (train first to get feature importance for LR) ──────────
message("Training Random Forest...")
set.seed(42)
rf_model <- ranger(
  x = X_train, y = y_train,
  num.trees = 500, mtry = floor(sqrt(ncol(X_train))),
  min.node.size = 10, importance = "impurity",
  probability = TRUE, seed = 42, classification = TRUE
)

# ── 2. LASSO ─────────────────────────────────────────────────────────────────
message("Training LASSO...")
set.seed(42)
cv_lasso <- cv.glmnet(
  x = as.matrix(X_train_sc), y = y_train,
  family = "binomial", alpha = 1, nfolds = 10, type.measure = "auc"
)

# ── 3. Logistic Regression (top 20 features by RF importance) ────────────────
message("Training Logistic Regression...")
imp_rf <- data.frame(
  feature    = names(rf_model$variable.importance),
  importance = rf_model$variable.importance
) %>% arrange(desc(importance))

top20_features <- imp_rf$feature[1:min(20, nrow(imp_rf))]

lr_train_df <- cbind(
  y = as.numeric(y_train == "Died"),
  X_train[, top20_features, drop = FALSE]
)
lr_model <- glm(y ~ ., data = lr_train_df, family = binomial())

# ── 4. Neural Network ─────────────────────────────────────────────────────────
message("Training Neural Network...")
set.seed(42)
nn_model <- nnet(
  x = as.matrix(X_train_sc),
  y = as.numeric(y_train == "Died"),
  size = 15, linout = FALSE, entropy = TRUE,
  decay = 0.01, maxit = 300, trace = FALSE,
  MaxNWts = 5000   # allow larger networks (default 1000 is too small for 100+ features)
)

message("All models trained.")

# ── Test-set predictions ──────────────────────────────────────────────────────
prob_lr_test <- predict(
  lr_model,
  newdata = cbind(y = 0, X_test[, top20_features, drop = FALSE]),
  type = "response"
)
prob_lasso_test <- predict(cv_lasso, newx = as.matrix(X_test_sc),
                           s = "lambda.1se", type = "response")[, 1]
prob_rf_test    <- predict(rf_model, data = X_test)$predictions[, "Died"]
prob_nn_test    <- as.vector(predict(nn_model, as.matrix(X_test_sc)))

# ── ROC objects ───────────────────────────────────────────────────────────────
roc_lr    <- roc(y_test, prob_lr_test,    levels = c("Survived", "Died"), direction = "<")
roc_lasso <- roc(y_test, prob_lasso_test, levels = c("Survived", "Died"), direction = "<")
roc_rf    <- roc(y_test, prob_rf_test,    levels = c("Survived", "Died"), direction = "<")
roc_nn    <- roc(y_test, prob_nn_test,    levels = c("Survived", "Died"), direction = "<")

# ── LASSO feature importances ─────────────────────────────────────────────────
coefs_lasso <- coef(cv_lasso, s = "lambda.1se")
imp_lasso <- data.frame(
  feature = rownames(coefs_lasso),
  coef    = as.numeric(coefs_lasso)
) %>%
  filter(feature != "(Intercept)", coef != 0) %>%
  arrange(desc(abs(coef)))

# ── Training set means (for LASSO SHAP approximation) ────────────────────────
lasso_features <- imp_lasso$feature
lasso_coefs    <- setNames(imp_lasso$coef, imp_lasso$feature)
train_means_sc <- colMeans(X_train_sc)  # all 0 after scaling by definition

# ── Performance metrics helper ────────────────────────────────────────────────
get_metrics <- function(roc_obj, probs, truth, model_name) {
  best  <- coords(roc_obj, "best", ret = "all", best.method = "youden")
  thresh <- as.numeric(best$threshold)[1]
  pred  <- factor(ifelse(probs >= thresh, "Died", "Survived"),
                  levels = c("Survived", "Died"))
  cm    <- confusionMatrix(pred, truth, positive = "Died")
  data.frame(
    Model       = model_name,
    AUROC       = round(auc(roc_obj), 3),
    Threshold   = round(thresh, 3),
    Sensitivity = round(cm$byClass["Sensitivity"], 3),
    Specificity = round(cm$byClass["Specificity"], 3),
    PPV         = round(cm$byClass["Pos Pred Value"], 3),
    NPV         = round(cm$byClass["Neg Pred Value"], 3),
    Accuracy    = round(cm$overall["Accuracy"], 3),
    Kappa       = round(cm$overall["Kappa"], 3)
  )
}

metrics_df <- bind_rows(
  get_metrics(roc_lr,    prob_lr_test,    y_test, "Logistic Regression"),
  get_metrics(roc_lasso, prob_lasso_test, y_test, "LASSO"),
  get_metrics(roc_rf,    prob_rf_test,    y_test, "Random Forest"),
  get_metrics(roc_nn,    prob_nn_test,    y_test, "Neural Network")
)

# ── Key input features for prediction calculator ─────────────────────────────
key_features <- c(
  "admission_age", "sofa_score",
  "heart_rate_mean", "sbp_mean", "mbp_mean", "resp_rate_mean",
  "temperature_mean", "spo2_mean",
  "creatinine_max", "bun_max", "albumin_min",
  "wbc_max", "platelets_min", "hemoglobin_min",
  "sodium_max", "potassium_max", "bicarbonate_min", "aniongap_max",
  "BAR", "BCR", "los_icu"
)

# Training means for centering in SHAP approximation
train_means <- colMeans(X_train, na.rm = TRUE)

# =============================================================================
# UI
# =============================================================================
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Sepsis 30-Day Mortality — MIMIC-IV (Demo)"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Cohort Explorer",    tabName = "cohort",     icon = icon("users")),
      menuItem("Patient Trajectory", tabName = "trajectory", icon = icon("chart-line")),
      menuItem("ML Prediction App",  tabName = "ml",         icon = icon("brain"))
    )
  ),
  dashboardBody(
    tags$div(
      style = "background:#fff3cd; padding:10px; border-left:4px solid #ffc107; margin-bottom:15px; font-size:13px;",
      HTML("<b>&#9888;&#65039; Demo Notice:</b> This public demo runs on <b>synthetic data</b> generated to mimic MIMIC-IV statistical properties. The original analysis used credentialed MIMIC-IV data (PhysioNet DUA). All patient identifiers shown are synthetic.")
    ),
    tabItems(

      # ── Tab 1: Cohort Explorer ───────────────────────────────────────────────
      tabItem(
        tabName = "cohort",
        fluidRow(
          valueBoxOutput("n_patients",     width = 3),
          valueBoxOutput("mortality_rate", width = 3),
          valueBoxOutput("median_age",     width = 3),
          valueBoxOutput("median_sofa",    width = 3)
        ),
        fluidRow(
          box(title = "Filters", status = "primary", solidHeader = TRUE, width = 3,
              sliderInput("age_range",  "Age Range:",    min = 18, max = 100, value = c(18, 100)),
              selectInput("gender_filter", "Gender:",    choices = c("All", "M", "F")),
              selectInput("race_filter",   "Race:",
                          choices = c("All", "White", "Black", "Asian", "Hispanic", "Other")),
              sliderInput("sofa_range", "SOFA Score:",  min = 0,  max = 24,  value = c(0, 24)),
              actionButton("reset_filters", "Reset Filters", icon = icon("refresh"))
          ),
          box(title = "Mortality by Age Group", status = "info", solidHeader = TRUE, width = 4,
              plotlyOutput("plot_age_mortality", height = "350px")),
          box(title = "Mortality by SOFA Category", status = "info", solidHeader = TRUE, width = 5,
              plotlyOutput("plot_sofa_mortality", height = "350px"))
        ),
        fluidRow(
          box(title = "Distribution of Key Variables", status = "warning", solidHeader = TRUE, width = 6,
              selectInput("dist_var", "Select Variable:",
                          choices = c("admission_age", "sofa_score", "los_icu",
                                      "heart_rate_mean", "sbp_mean", "creatinine_max",
                                      "bun_max", "albumin_min", "BAR", "BCR",
                                      "charlson_comorbidity_index")),
              plotlyOutput("plot_distribution", height = "300px")),
          box(title = "Cohort Data Table", status = "success", solidHeader = TRUE, width = 6,
              DTOutput("cohort_table", height = "350px"))
        )
      ),

      # ── Tab 2: Patient Trajectory ─────────────────────────────────────────────
      tabItem(
        tabName = "trajectory",
        fluidRow(
          box(title = "Select Patient", status = "primary", solidHeader = TRUE, width = 3,
              numericInput("patient_id", "Subject ID:",
                           value = df_raw$subject_id[1], min = 1),
              actionButton("find_patient", "Load Patient", icon = icon("search")),
              hr(),
              h4("Patient Summary"),
              uiOutput("patient_summary")
          ),
          box(title = "Vital Signs — First 24h Ranges", status = "info", solidHeader = TRUE, width = 9,
              plotlyOutput("vital_radar", height = "400px"))
        ),
        fluidRow(
          box(title = "Lab Results — First 24h (Min / Max)", status = "warning", solidHeader = TRUE, width = 6,
              plotlyOutput("lab_bars", height = "350px")),
          box(title = "SOFA Component Breakdown", status = "danger", solidHeader = TRUE, width = 6,
              plotlyOutput("sofa_breakdown", height = "350px"))
        )
      ),

      # ── Tab 3: ML Prediction App ──────────────────────────────────────────────
      tabItem(
        tabName = "ml",
        fluidRow(
          box(title = "Model Performance Comparison — Test Set",
              status = "primary", solidHeader = TRUE, width = 12,
              fluidRow(
                column(7, plotlyOutput("roc_plot", height = "420px")),
                column(5, plotlyOutput("importance_plot", height = "420px"))
              )
          )
        ),
        fluidRow(
          box(title = "Performance Metrics Table",
              status = "info", solidHeader = TRUE, width = 12,
              DTOutput("metrics_table_global"))
        ),
        fluidRow(
          box(title = "Patient Risk Calculator", status = "danger", solidHeader = TRUE, width = 4,
              h4("Enter First-24h Patient Values"),
              numericInput("calc_age",   "Age:",               value = 65,  min = 18,  max = 100),
              numericInput("calc_sofa",  "SOFA Score:",        value = 6,   min = 0,   max = 24),
              numericInput("calc_hr",    "Heart Rate (mean):", value = 90,  min = 30,  max = 200),
              numericInput("calc_sbp",   "SBP (mean):",        value = 110, min = 40,  max = 220),
              numericInput("calc_mbp",   "MBP (mean):",        value = 70,  min = 20,  max = 150),
              numericInput("calc_rr",    "Resp Rate (mean):",  value = 20,  min = 5,   max = 50),
              numericInput("calc_temp",  "Temperature (mean):", value = 37.0, min = 32, max = 42, step = 0.1),
              numericInput("calc_spo2",  "SpO\u2082 (mean):", value = 96,  min = 50,  max = 100),
              numericInput("calc_creat", "Creatinine (max):",  value = 1.2, min = 0.1, max = 20,  step = 0.1),
              numericInput("calc_bun",   "BUN (max):",         value = 25,  min = 1,   max = 200),
              numericInput("calc_alb",   "Albumin (min):",     value = 3.5, min = 0.5, max = 6.0, step = 0.1),
              numericInput("calc_wbc",   "WBC (max):",         value = 12,  min = 0.1, max = 100),
              numericInput("calc_plt",   "Platelets (min):",   value = 200, min = 1,   max = 1000),
              numericInput("calc_hgb",   "Hemoglobin (min):",  value = 10,  min = 2,   max = 20,  step = 0.1),
              numericInput("calc_na",    "Sodium (max):",      value = 140, min = 110, max = 170),
              numericInput("calc_k",     "Potassium (max):",   value = 4.0, min = 2.0, max = 8.0, step = 0.1),
              numericInput("calc_bicarb","Bicarbonate (min):", value = 22,  min = 5,   max = 40),
              numericInput("calc_ag",    "Anion Gap (max):",   value = 14,  min = 1,   max = 40),
              numericInput("calc_los",   "ICU LOS (days):",    value = 3,   min = 0,   max = 60,  step = 0.1),
              actionButton("predict_btn", "Predict 30-Day Risk",
                           icon = icon("calculator"), class = "btn-danger btn-lg btn-block")
          ),
          box(title = "Prediction Results — All Four Models",
              status = "success", solidHeader = TRUE, width = 8,
              uiOutput("prediction_results"),
              hr(),
              h4("LASSO Risk Factor Contributions (linear SHAP)"),
              p(style = "color:#666; font-size:0.9em;",
                "Each bar shows how much this feature pushes predicted risk up (red) or down (green) relative to the population average. This is the exact SHAP decomposition for a linear model."),
              plotlyOutput("shap_waterfall", height = "380px")
          )
        )
      )
    )
  )
)

# =============================================================================
# SERVER
# =============================================================================
server <- function(input, output, session) {

  # ── Reactive filtered data ──────────────────────────────────────────────────
  filtered_df <- reactive({
    d <- df %>%
      filter(
        admission_age >= input$age_range[1],
        admission_age <= input$age_range[2],
        sofa_score    >= input$sofa_range[1],
        sofa_score    <= input$sofa_range[2]
      )
    if (input$gender_filter != "All") d <- d %>% filter(gender == input$gender_filter)
    if (input$race_filter   != "All") d <- d %>% filter(race_group == input$race_filter)
    d
  })

  observeEvent(input$reset_filters, {
    updateSliderInput(session, "age_range",  value = c(18, 100))
    updateSliderInput(session, "sofa_range", value = c(0, 24))
    updateSelectInput(session, "gender_filter", selected = "All")
    updateSelectInput(session, "race_filter",   selected = "All")
  })

  # ── Value boxes ─────────────────────────────────────────────────────────────
  output$n_patients <- renderValueBox({
    valueBox(format(nrow(filtered_df()), big.mark = ","), "Patients",
             icon = icon("users"), color = "blue")
  })
  output$mortality_rate <- renderValueBox({
    rate <- round(mean(filtered_df()$label_30d == "Died") * 100, 1)
    valueBox(paste0(rate, "%"), "30-Day Mortality",
             icon = icon("heartbeat"), color = "red")
  })
  output$median_age <- renderValueBox({
    valueBox(round(median(filtered_df()$admission_age, na.rm = TRUE), 0), "Median Age",
             icon = icon("birthday-cake"), color = "yellow")
  })
  output$median_sofa <- renderValueBox({
    valueBox(round(median(filtered_df()$sofa_score, na.rm = TRUE), 1), "Median SOFA",
             icon = icon("chart-bar"), color = "purple")
  })

  # ── Cohort plots ─────────────────────────────────────────────────────────────
  output$plot_age_mortality <- renderPlotly({
    d <- filtered_df() %>%
      group_by(age_group, label_30d) %>% summarise(n = n(), .groups = "drop") %>%
      group_by(age_group) %>% mutate(pct = round(100 * n / sum(n), 1))
    p <- ggplot(d, aes(x = age_group, y = pct, fill = label_30d)) +
      geom_col(position = "dodge") +
      scale_fill_manual(values = c("Survived" = "#2196F3", "Died" = "#F44336")) +
      labs(x = "Age Group", y = "Percentage (%)", fill = "Outcome") +
      theme_minimal(base_size = 13)
    ggplotly(p)
  })

  output$plot_sofa_mortality <- renderPlotly({
    d <- filtered_df() %>%
      group_by(sofa_cat, label_30d) %>% summarise(n = n(), .groups = "drop") %>%
      group_by(sofa_cat) %>% mutate(pct = round(100 * n / sum(n), 1))
    p <- ggplot(d, aes(x = sofa_cat, y = pct, fill = label_30d)) +
      geom_col(position = "dodge") +
      scale_fill_manual(values = c("Survived" = "#2196F3", "Died" = "#F44336")) +
      labs(x = "SOFA Category", y = "Percentage (%)", fill = "Outcome") +
      theme_minimal(base_size = 13)
    ggplotly(p)
  })

  output$plot_distribution <- renderPlotly({
    d   <- filtered_df()
    var <- input$dist_var
    if (!var %in% names(d)) return(plotly_empty())
    p <- ggplot(d, aes(x = .data[[var]], fill = label_30d)) +
      geom_density(alpha = 0.5) +
      scale_fill_manual(values = c("Survived" = "#2196F3", "Died" = "#F44336")) +
      labs(x = var, y = "Density", fill = "Outcome") +
      theme_minimal(base_size = 13)
    ggplotly(p)
  })

  output$cohort_table <- renderDT({
    filtered_df() %>%
      select(subject_id, admission_age, gender, race_group, sofa_score,
             charlson_comorbidity_index, bun_max, albumin_min,
             BAR, BCR, los_icu, label_30d) %>%
      slice_head(n = 500) %>%
      datatable(options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
  })

  # ── Tab 2: Patient Trajectory ─────────────────────────────────────────────
  patient_data <- reactiveVal(NULL)

  observeEvent(input$find_patient, {
    pt <- df_raw %>% filter(subject_id == input$patient_id) %>% slice(1)
    if (nrow(pt) > 0) patient_data(pt) else patient_data(NULL)
  })

  output$patient_summary <- renderUI({
    pt <- patient_data()
    if (is.null(pt)) return(tags$p("Enter a Subject ID and click Load."))
    outcome_style <- if (pt$label_30d == 1) "color:red; font-weight:bold;" else "color:green; font-weight:bold;"
    tags$div(
      tags$p(tags$b("Age: "), pt$admission_age),
      tags$p(tags$b("Gender: "), pt$gender),
      tags$p(tags$b("Race: "), pt$race),
      tags$p(tags$b("SOFA: "), pt$sofa_score),
      tags$p(tags$b("CCI: "), round(pt$charlson_comorbidity_index, 1)),
      tags$p(tags$b("ICU LOS: "), round(pt$los_icu, 1), " days"),
      tags$p(tags$b("BAR: "), round(pt$BAR, 2)),
      tags$p(tags$b("BCR: "), round(pt$BCR, 2)),
      tags$p(tags$b("30-Day Outcome: "),
             tags$span(style = outcome_style,
                       ifelse(pt$label_30d == 1, "Died", "Survived")))
    )
  })

  output$vital_radar <- renderPlotly({
    pt <- patient_data()
    if (is.null(pt)) return(plotly_empty())
    vitals <- data.frame(
      Vital = c("HR", "SBP", "DBP", "MBP", "RR", "Temp", "SpO\u2082"),
      Min   = as.numeric(c(pt$heart_rate_min, pt$sbp_min, pt$dbp_min, pt$mbp_min,
                            pt$resp_rate_min, pt$temperature_min, pt$spo2_min)),
      Mean  = as.numeric(c(pt$heart_rate_mean, pt$sbp_mean, pt$dbp_mean, pt$mbp_mean,
                            pt$resp_rate_mean, pt$temperature_mean, pt$spo2_mean)),
      Max   = as.numeric(c(pt$heart_rate_max, pt$sbp_max, pt$dbp_max, pt$mbp_max,
                            pt$resp_rate_max, pt$temperature_max, pt$spo2_max))
    )
    plot_ly(vitals, x = ~Vital, y = ~Mean, type = "bar", name = "Mean",
            marker = list(color = "#2196F3"),
            error_y = list(type = "data", symmetric = FALSE,
                           array       = vitals$Max - vitals$Mean,
                           arrayminus  = vitals$Mean - vitals$Min,
                           color = "#333")) %>%
      layout(title = "Vital Signs — Min/Mean/Max (First 24h)",
             yaxis = list(title = "Value"), xaxis = list(title = ""))
  })

  output$lab_bars <- renderPlotly({
    pt <- patient_data()
    if (is.null(pt)) return(plotly_empty())
    labs_df <- data.frame(
      Lab = c("Creatinine", "BUN", "Albumin", "WBC", "Hgb", "Plt",
              "Na", "K", "HCO\u2083", "Anion Gap"),
      Min = as.numeric(c(pt$creatinine_min, pt$bun_min, pt$albumin_min,
                          pt$wbc_min, pt$hemoglobin_min, pt$platelets_min,
                          pt$sodium_min, pt$potassium_min,
                          pt$bicarbonate_min, pt$aniongap_min)),
      Max = as.numeric(c(pt$creatinine_max, pt$bun_max, pt$albumin_max,
                          pt$wbc_max, pt$hemoglobin_max, pt$platelets_max,
                          pt$sodium_max, pt$potassium_max,
                          pt$bicarbonate_max, pt$aniongap_max))
    ) %>%
      pivot_longer(cols = c(Min, Max), names_to = "stat", values_to = "value")
    plot_ly(labs_df, x = ~Lab, y = ~value, color = ~stat, type = "bar",
            colors = c("Min" = "#FF9800", "Max" = "#F44336")) %>%
      layout(title = "Lab Values — Min/Max (First 24h)", barmode = "group",
             yaxis = list(title = "Value"), xaxis = list(title = ""))
  })

  output$sofa_breakdown <- renderPlotly({
    pt <- patient_data()
    if (is.null(pt)) return(plotly_empty())
    pick <- function(y_col, x_col) {
      val_y <- if (!is.null(pt[[y_col]]) && !is.na(pt[[y_col]])) pt[[y_col]] else NA
      val_x <- if (!is.null(pt[[x_col]]) && !is.na(pt[[x_col]])) pt[[x_col]] else NA
      ifelse(!is.na(val_y), val_y, ifelse(!is.na(val_x), val_x, 0))
    }
    sofa_df <- data.frame(
      Component = c("Respiration", "Coagulation", "Liver", "Cardiovascular", "CNS", "Renal"),
      Score     = as.numeric(c(
        pick("respiration_y", "respiration_x"),
        pick("coagulation_y", "coagulation_x"),
        pick("liver_y",       "liver_x"),
        pick("cardiovascular_y", "cardiovascular_x"),
        pick("cns_y",         "cns_x"),
        pick("renal_y",       "renal_x")
      ))
    )
    color_map <- c("#4CAF50", "#FF9800", "#F44336", "#9C27B0")
    sofa_df$color <- color_map[pmin(pmax(sofa_df$Score + 1, 1), 4)]
    plot_ly(sofa_df, x = ~Component, y = ~Score, type = "bar",
            marker = list(color = sofa_df$color)) %>%
      layout(title = paste0("SOFA Components (Total: ", pt$sofa_score, ")"),
             yaxis = list(title = "Score (0–4)", range = c(0, 4.5)),
             xaxis = list(title = ""))
  })

  # ── Tab 3: ROC Comparison ─────────────────────────────────────────────────
  output$roc_plot <- renderPlotly({
    make_roc_df <- function(roc_obj, name) {
      data.frame(
        fpr   = 1 - roc_obj$specificities,
        tpr   = roc_obj$sensitivities,
        model = name
      ) %>% arrange(fpr)
    }
    model_cols <- c(
      "LR (AUC)"    = "#FF9800",
      "LASSO (AUC)" = "#F44336",
      "RF (AUC)"    = "#3F51B5",
      "NN (AUC)"    = "#009688"
    )
    roc_list <- list(
      list(roc_lr,    paste0("LR (AUC=",    round(auc(roc_lr),    3), ")"), "#FF9800"),
      list(roc_lasso, paste0("LASSO (AUC=", round(auc(roc_lasso), 3), ")"), "#F44336"),
      list(roc_rf,    paste0("RF (AUC=",    round(auc(roc_rf),    3), ")"), "#3F51B5"),
      list(roc_nn,    paste0("NN (AUC=",    round(auc(roc_nn),    3), ")"), "#009688")
    )
    p <- plot_ly()
    for (m in roc_list) {
      df_roc <- data.frame(fpr = 1 - m[[1]]$specificities, tpr = m[[1]]$sensitivities) %>%
        arrange(fpr)
      p <- add_lines(p, data = df_roc, x = ~fpr, y = ~tpr,
                     name = m[[2]], line = list(color = m[[3]], width = 2))
    }
    p %>%
      add_lines(x = c(0, 1), y = c(0, 1), name = "Reference",
                line = list(color = "grey", dash = "dash", width = 1)) %>%
      layout(title = "ROC Curves — Test Set (All 4 Models)",
             xaxis = list(title = "1 - Specificity (FPR)"),
             yaxis = list(title = "Sensitivity (TPR)"),
             legend = list(x = 0.5, y = 0.1))
  })

  output$importance_plot <- renderPlotly({
    top_rf <- imp_rf %>% slice_head(n = 15)
    plot_ly(top_rf, x = ~importance, y = ~reorder(feature, importance),
            type = "bar", orientation = "h",
            marker = list(color = "#3F51B5")) %>%
      layout(title = "Top 15 RF Feature Importance",
             xaxis = list(title = "Impurity Importance"),
             yaxis = list(title = ""))
  })

  output$metrics_table_global <- renderDT({
    metrics_df %>%
      datatable(
        options = list(dom = 't', paging = FALSE, scrollX = TRUE),
        rownames = FALSE
      ) %>%
      formatStyle("AUROC", fontWeight = "bold",
                  backgroundColor = styleInterval(c(0.70, 0.80),
                                                  c("#ffcccc", "#ffe0b2", "#c8e6c9")))
  })

  # ── Prediction Calculator ─────────────────────────────────────────────────
  pred_results <- reactiveVal(NULL)

  observeEvent(input$predict_btn, {

    # Build input row matching the full feature matrix
    new_row <- X_mat[1, , drop = FALSE]
    new_row[1, ] <- 0  # zero out

    # Map UI inputs to feature names
    fill_if <- function(feat, val) {
      if (feat %in% names(new_row)) new_row[[feat]] <<- val
    }
    fill_if("admission_age",    input$calc_age)
    fill_if("sofa_score",       input$calc_sofa)
    fill_if("sofa",             input$calc_sofa)
    fill_if("heart_rate_mean",  input$calc_hr)
    fill_if("heart_rate_min",   input$calc_hr)
    fill_if("heart_rate_max",   input$calc_hr)
    fill_if("sbp_mean",         input$calc_sbp)
    fill_if("sbp_min",          input$calc_sbp)
    fill_if("sbp_max",          input$calc_sbp)
    fill_if("mbp_mean",         input$calc_mbp)
    fill_if("mbp_min",          input$calc_mbp)
    fill_if("mbp_max",          input$calc_mbp)
    fill_if("resp_rate_mean",   input$calc_rr)
    fill_if("resp_rate_min",    input$calc_rr)
    fill_if("resp_rate_max",    input$calc_rr)
    fill_if("temperature_mean", input$calc_temp)
    fill_if("temperature_min",  input$calc_temp)
    fill_if("temperature_max",  input$calc_temp)
    fill_if("spo2_mean",        input$calc_spo2)
    fill_if("spo2_min",         input$calc_spo2)
    fill_if("spo2_max",         input$calc_spo2)
    fill_if("creatinine_max",   input$calc_creat)
    fill_if("creatinine_min",   input$calc_creat)
    fill_if("bun_max",          input$calc_bun)
    fill_if("bun_min",          input$calc_bun)
    fill_if("albumin_min",      input$calc_alb)
    fill_if("albumin_max",      input$calc_alb)
    fill_if("wbc_max",          input$calc_wbc)
    fill_if("wbc_min",          input$calc_wbc)
    fill_if("platelets_min",    input$calc_plt)
    fill_if("platelets_max",    input$calc_plt)
    fill_if("hemoglobin_min",   input$calc_hgb)
    fill_if("hemoglobin_max",   input$calc_hgb)
    fill_if("sodium_max",       input$calc_na)
    fill_if("sodium_min",       input$calc_na)
    fill_if("potassium_max",    input$calc_k)
    fill_if("potassium_min",    input$calc_k)
    fill_if("bicarbonate_min",  input$calc_bicarb)
    fill_if("bicarbonate_max",  input$calc_bicarb)
    fill_if("aniongap_max",     input$calc_ag)
    fill_if("aniongap_min",     input$calc_ag)
    fill_if("los_icu",          input$calc_los)
    # Derived ratios
    BAR_val <- input$calc_bun / max(input$calc_alb, 0.1)
    BCR_val <- input$calc_bun / max(input$calc_creat, 0.01)
    fill_if("BAR", BAR_val)
    fill_if("BCR", BCR_val)

    # Scale
    new_sc <- scale(new_row, center = sc_center, scale = sc_scale)
    new_sc[!is.finite(new_sc)] <- 0

    # LR prediction
    lr_row <- new_row[, top20_features, drop = FALSE]
    lr_row_d <- cbind(y = 0, lr_row)
    prob_l <- as.numeric(predict(lr_model, newdata = lr_row_d, type = "response"))

    # LASSO prediction
    prob_la <- as.numeric(predict(cv_lasso, newx = as.matrix(new_sc),
                                  s = "lambda.1se", type = "response")[1, 1])

    # RF prediction
    prob_r <- predict(rf_model, data = new_row)$predictions[1, "Died"]

    # NN prediction
    prob_n <- as.numeric(predict(nn_model, as.matrix(new_sc))[1])

    # ── LASSO linear SHAP (exact for linear models) ──────────────────────────
    # φ_i = β_i × (x_i_scaled - 0)  [train means = 0 after scaling]
    shap_feats   <- intersect(lasso_features, colnames(new_sc))
    shap_contrib <- lasso_coefs[shap_feats] * as.numeric(new_sc[1, shap_feats])
    shap_df      <- data.frame(
      feature     = shap_feats,
      contribution = shap_contrib
    ) %>%
      arrange(desc(abs(contribution))) %>%
      slice_head(n = 15)

    pred_results(list(
      prob_lr    = prob_l,
      prob_lasso = prob_la,
      prob_rf    = prob_r,
      prob_nn    = prob_n,
      shap_df    = shap_df
    ))
  })

  output$prediction_results <- renderUI({
    res <- pred_results()
    if (is.null(res)) {
      return(tags$p(style = "color:#999; text-align:center; margin-top:30px;",
                    "Enter patient values and click 'Predict 30-Day Risk'"))
    }

    risk_color <- function(p) {
      if (p > 0.35) "#F44336" else if (p > 0.20) "#FF9800" else "#4CAF50"
    }
    risk_label <- function(p) {
      if (p > 0.35) "HIGH RISK" else if (p > 0.20) "MODERATE RISK" else "LOW RISK"
    }

    make_card <- function(name, prob) {
      col <- risk_color(prob)
      tags$div(
        style = paste0("text-align:center; padding:15px; border:2px solid ", col,
                       "; border-radius:10px; margin:8px;"),
        tags$h4(name),
        tags$h2(style = paste0("color:", col, ";"), paste0(round(prob * 100, 1), "%")),
        tags$b(style  = paste0("color:", col, ";"), risk_label(prob))
      )
    }

    tagList(
      fluidRow(
        column(3, make_card("Logistic Regression", res$prob_lr)),
        column(3, make_card("LASSO",               res$prob_lasso)),
        column(3, make_card("Random Forest",        res$prob_rf)),
        column(3, make_card("Neural Network",       res$prob_nn))
      ),
      tags$p(
        style = "text-align:center; color:#666; margin-top:10px; font-size:0.9em;",
        paste0("Ensemble mean: ",
               round(mean(c(res$prob_lr, res$prob_lasso, res$prob_rf, res$prob_nn)) * 100, 1),
               "% | Risk thresholds: LOW < 20% | MODERATE 20–35% | HIGH > 35%")
      )
    )
  })

  output$shap_waterfall <- renderPlotly({
    res <- pred_results()
    if (is.null(res)) return(plotly_empty())

    df_shap <- res$shap_df %>%
      arrange(contribution) %>%
      mutate(
        color   = ifelse(contribution > 0, "#F44336", "#4CAF50"),
        feature = factor(feature, levels = feature)
      )

    plot_ly(df_shap,
            x = ~contribution, y = ~feature, type = "bar", orientation = "h",
            marker = list(color = df_shap$color)) %>%
      add_lines(x = c(0, 0), y = c(-0.5, nrow(df_shap) - 0.5),
                line = list(color = "black", width = 1), showlegend = FALSE) %>%
      layout(
        title  = "LASSO Feature Contributions (β × x_scaled)",
        xaxis  = list(title = "Risk contribution (positive = higher risk)"),
        yaxis  = list(title = ""),
        margin = list(l = 180)
      )
  })
}

# Run app
shinyApp(ui, server)

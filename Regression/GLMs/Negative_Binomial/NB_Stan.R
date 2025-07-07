# ------------------------------------------------------------
# 0. Pacotes  (=> trocamos **rstan** por **cmdstanr**)
# ------------------------------------------------------------
library(tidyverse)
library(cmdstanr)                 # <- mudança principal
library(posterior)
# se ainda não tiver o CmdStan instalado, faça uma única vez:
# cmdstanr::install_cmdstan()

# ------------------------------------------------------------
# 1. Dados  (mesma lógica que antes)
# ------------------------------------------------------------
url <- "https://raw.githubusercontent.com/COINtoolbox/NB_GCs/refs/heads/master/Dataset/GCs.csv"

GCS <- read.csv(url, header = TRUE, dec = ".", sep = "") |>
  drop_na(Mdyn)

N     <- nrow(GCS)
L_raw <- pmax(GCS$N_GC - GCS$N_GC_err, 0)
U     <- GCS$N_GC + GCS$N_GC_err
Lm1   <- pmax(L_raw - 1, 0)

MBH_obs   <- GCS$MBH
MBH_sigma <- GCS$upMBH

MBHx <- seq(0.95 * min(MBH_obs),
            1.05 * max(MBH_obs), length.out = 500)

stan_data <- list(
  N        = N,
  L        = L_raw,
  Lm1      = Lm1,
  U        = U,
  MBH_obs  = MBH_obs,
  MBH_sigma= MBH_sigma,
  M        = length(MBHx),
  MBHx     = MBHx
)

stan_code <- "
data {
  int<lower=1>  N;
  array[N] int<lower=0>  L;      // <- NOVA sintaxe
  array[N] int<lower=0>  Lm1;
  array[N] int<lower=0>  U;
  vector[N]     MBH_obs;
  vector<lower=0>[N] MBH_sigma;
  int<lower=1>  M;
  vector[M]     MBHx;
}
parameters {
  real          beta0;
  real          beta1;
  real<lower=0> phi;
  vector[N]     MBH_true;
}
transformed parameters {
  vector[N] mu = exp(beta0 + beta1 .* MBH_true);
}
model {
  beta0    ~ normal(0, 10);
  beta1    ~ normal(0, 10);
  phi      ~ gamma(2, 0.1);
  MBH_obs  ~ normal(MBH_true, MBH_sigma);

  for (i in 1:N)
    target += neg_binomial_2_lcdf(U[i]  | mu[i], phi)
            - neg_binomial_2_lcdf(Lm1[i] | mu[i], phi);
}
generated quantities {
  vector[M] mu_pred;
  array[M] int prediction_NBx;

  for (j in 1:M) {
    real eta = beta0 + beta1 * MBHx[j];
    real capped_eta = fmin(20, fmax(-20, eta));
    mu_pred[j] = exp(capped_eta);
    prediction_NBx[j] = neg_binomial_2_rng(mu_pred[j], phi);  // NB sample
  }
}
"


# ------------------------------------------------------------
# 3. Compilação (apenas a 1ª vez; depois fica em cache)
# ------------------------------------------------------------
model_file <- write_stan_file(stan_code)   # cria .stan temporário
mod        <- cmdstan_model(model_file)

# ------------------------------------------------------------
# 4. Amostragem  (4 cadeias paralelas + tunings sugeridos)
# ------------------------------------------------------------


lm_start <- lm(log1p(N_GC) ~ MBH, data = GCS)        # log1p evita log(0)
beta1_hat <- coef(lm_start)[2]
beta0_hat <- coef(lm_start)[1]

init_fun <- function()list(
  beta0     = beta0_hat,     # ~ –4.7
  beta1     = beta1_hat,     # ~ 1.2
  phi       = 10,            # começa com dispersão razoável
  MBH_true  = MBH_obs        # centrado na observação
)


fit <- mod$sample(
  data            = stan_data,
  init            = init_fun,
  chains          = 1,
  parallel_chains = 1,
  iter_warmup     = 500,
  iter_sampling   = 1500,
  seed            = 42,
  adapt_delta     = 0.95,      # eleve para 0.95 se surgirem divergências
  max_treedepth   = 12,
  refresh         = 500
)

# ver diagnóstico rápido
fit$cmdstan_diagnose()

# ------------------------------------------------------------
# 5. Resumo para plot (mudou só a maneira de extrair)
# ------------------------------------------------------------


ci <- fit$draws("prediction_NBx") |>               # draw_array → posterior
  as_draws_matrix() |>
  apply(2, quantile, probs = c(.025, .5, .975)) |>
  t() |>
  as_tibble(.name_repair = "minimal") |>
  setNames(c("lwr", "mid", "upr")) |>
  mutate(MBHx = MBHx)

# ------------------------------------------------------------
# 6. Plot (idêntico ao seu)
# ------------------------------------------------------------
ggplot() +
  geom_ribbon(data = ci,
              aes(x = MBHx, ymin = lwr, ymax = upr),
              fill = "grey70", alpha = .25) +
  geom_line(data = ci,
            aes(x = MBHx, y = mid),
            linewidth = 1.1, linetype = "dashed", colour = "black") +
  geom_point(data = GCS,
             aes(x = MBH, y = N_GC, colour = Type, shape = Type),
             size = 3, alpha = .85) +
  geom_errorbar(data = GCS,
                aes(x = MBH,
                    ymin = pmax(N_GC - N_GC_err, 0),
                    ymax = N_GC + N_GC_err,
                    colour = Type),
                width = .05, alpha = .7) +
  geom_errorbarh(data = GCS,
                 aes(y = N_GC,
                     xmin = MBH - lowMBH,
                     xmax = MBH + upMBH,
                     colour = Type),
                 height = .05, alpha = .7) +
  scale_y_continuous(trans = scales::asinh_trans(),
                     breaks = c(0, 10, 100, 1e3, 1e4, 1e5),
                     labels = c("0", expression(10^1), expression(10^2),
                                expression(10^3), expression(10^4), expression(10^5))) +
  labs(x = expression(log~M[BH]/M['\u0298']),
       y = expression(N[GC]),
       colour = NULL) +
  theme_bw(base_size = 16) +
  theme(legend.position = "top")


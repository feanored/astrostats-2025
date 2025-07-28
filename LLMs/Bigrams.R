# -------------------------
# 1. Corpus: Fatos históricos e culturais
# -------------------------
frases <- c(
  "Einstein desenvolveu a teoria da relatividade",
  "Darwin propôs a teoria da evolução",
  "Newton formulou as leis do movimento",
  "Mozart compôs dezenas de sinfonias",
  "Leonardo da Vinci pintou a Mona Lisa",
  "Marie Curie descobriu o rádio e o polônio",
  "Galileu observou luas de Júpiter com telescópio",
  "Shakespeare escreveu Hamlet e Macbeth",
  "Tarsila pintou o Abaporu no modernismo brasileiro",
  "Copérnico sugeriu que a Terra orbita o Sol"
)

# -------------------------
# 2. Tokenização
# -------------------------
palavras <- tolower(unlist(strsplit(frases, " ")))
vocab <- unique(palavras)
vocab_size <- length(vocab)

# -------------------------
# 3. Bigramas
# -------------------------
bigrams <- data.frame(contexto = NA, alvo = NA)
for (frase in frases) {
  tokens <- tolower(unlist(strsplit(frase, " ")))
  for (i in 1:(length(tokens) - 1)) {
    bigrams <- rbind(bigrams, data.frame(contexto = tokens[i], alvo = tokens[i + 1]))
  }
}
bigrams <- na.omit(bigrams)

# -------------------------
# 4. Modelo de Bigramas
# -------------------------
library(dplyr)
modelo_bigram <- bigrams %>%
  count(contexto, alvo) %>%
  group_by(contexto) %>%
  mutate(prob = n / sum(n)) %>%
  ungroup()

# -------------------------
# 5. Prever próxima palavra
# -------------------------
prever_proxima <- function(palavra) {
  opcoes <- modelo_bigram %>% filter(contexto == palavra)
  if (nrow(opcoes) == 0) return("...")
  sample(opcoes$alvo, 1, prob = opcoes$prob)
}

# -------------------------
# 6. Gerador de frases
# -------------------------
gerar_frase <- function(inicio = "a", max_palavras = 12) {
  atual <- inicio
  frase <- atual
  for (i in 2:max_palavras) {
    proxima <- prever_proxima(atual)
    if (proxima == "...") break
    frase <- paste(frase, proxima)
    atual <- proxima
  }
  return(frase)
}

set.seed(42)
cat("Frase gerada:\n", gerar_frase("a"), "\n\n")

# -------------------------
# 7. Embeddings one-hot
# -------------------------
embedding <- diag(vocab_size)
rownames(embedding) <- vocab

# -------------------------
# 8. Sistema de Perguntas e Respostas
# -------------------------
respostas <- frases

palavras_chave <- list(
  "quem desenvolveu" = "desenvolveu",
  "quem propôs" = "propôs",
  "quem formulou" = "formulou",
  "quem compôs" = "compôs",
  "quem pintou" = "pintou",
  "quem descobriu" = "descobriu",
  "quem observou" = "observou",
  "quem escreveu" = "escreveu",
  "quem sugeriu" = "sugeriu",
  "quem usou telescópio" = "telescópio"
)

responder <- function(pergunta) {
  pergunta <- tolower(pergunta)
  for (chave in names(palavras_chave)) {
    if (grepl(chave, pergunta)) {
      termo <- palavras_chave[[chave]]
      candidatos <- grep(termo, respostas, value = TRUE)
      if (length(candidatos) > 0) {
        return(sample(candidatos, 1))
      }
    }
  }
  return("Desculpe, não sei responder isso ainda.")
}

# Exemplos:
cat("Q: quem escreveu Hamlet?\nA:", responder("quem escreveu?"), "\n")
cat("Q: quem descobriu o rádio?\nA:", responder("quem descobriu?"), "\n")
cat("Q: quem sugeriu que a Terra orbita o Sol?\nA:", responder("quem sugeriu?"), "\n\n")

# -------------------------
# 9. Visualização dos embeddings
# -------------------------
library(ggplot2)
library(Rtsne)

palavras_escolhidas <- c("einstein", "darwin", "newton", "mozart", "Leonardo da Vinci",
                         "curie", "galileu", "shakespeare", "tarsila", "copérnico")

# Remove nomes que não estão no vocab por conta de acento ou minúscula
palavras_escolhidas <- palavras_escolhidas[palavras_escolhidas %in% rownames(embedding)]

matriz_emb <- embedding[palavras_escolhidas, ]

set.seed(123)
reduzido <- Rtsne(matriz_emb, dims = 2, perplexity = 1, verbose = FALSE)$Y
df_plot <- data.frame(reduzido, palavra = palavras_escolhidas)

ggplot(df_plot, aes(x = X1, y = X2, label = palavra)) +
  geom_point(size = 3, color = "darkred") +
  geom_text(vjust = -0.5, size = 5) +
  theme_minimal() +
  ggtitle("Visualização de embeddings one-hot com nomes históricos")


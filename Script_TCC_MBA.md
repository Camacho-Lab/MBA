# MBA
Código utilizado no TCC do MBA em Data Science e Analytics da USP/Esalq

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:55:01 2026

@author: Cleber Camacho
"""

#%% Bibliotecas utilizadas no projeto

# pip install pandas
# pip install numpy
# pip install scipy
# pip install statsmodels
# pip install scikit-learn
# pip install openpyxl
# pip install statstests
# pip install scikeras
# pip install tensorflow

#%% Importando as bibliotecas
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, chi2, chi2_contingency, fisher_exact
from statstests.process import stepwise
from statstests.tests import shapiro_francia
import statsmodels.api as sm

from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#%% Diretórios de saída dos dados

# Pasta base do projeto
PASTA_SAIDA = Path.cwd() / "resultados_MBA_Modelos_progressao"
PASTA_IMAGENS = PASTA_SAIDA / "imagens"
PASTA_TABELAS = PASTA_SAIDA / "tabelas"

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

timestamp_execucao = datetime.now().strftime("%Y%m%d_%H%M%S")

arquivo_excel_saida = PASTA_TABELAS / f"Resultados_MBA_Modelos_Progressao_{timestamp_execucao}.xlsx"

print("Diretório principal:", PASTA_SAIDA)
print("Diretório de imagens:", PASTA_IMAGENS)
print("Arquivo Excel:", arquivo_excel_saida)

#%% Carregando do arquivo Excel

dados = pd.read_excel("Dados_Integrados_CMT_RET_Final_130426.xlsx")

#%% Data Wrangling (mantendo o tempo de seguimento)

cols_drop = [
    "Nome do participante", "Peso (kg)", "Tamanho tumoral", "Tempo de duplicação < 24 meses",
    "Crescimento tumoral ≥ 50%", "Nova metástase", "Progressão da doença"
]

dados = dados.drop(columns=[c for c in cols_drop if c in dados.columns], errors="ignore")

rename_map = {
    "Sexo": "Gênero",
    "Seguimento": "Tempo de seguimento (anos)",
    "RET": "Rastreamento do gene RET",
    "Estadiamento (Tamanho)": "Estadiamento (tamanho)",
    "Estadiamento (linfonodos)": "Estadiamento (linfonodo)",
    "Ca Diferenciado": "Presença de Carcinoma Diferenciado",
    "Calcitonina pós-operatória": "Calcitonina sérica (pg/mL)"
}

dados = dados.rename(columns=rename_map)
dados = dados.dropna()

#%% Ajustes das variáveis e legendas
 
# 1) Gênero
if "Gênero" in dados.columns:
    dados["Gênero"] = dados["Gênero"].astype(str).str.strip().str.lower()
    dados.loc[dados["Gênero"].isin(["feminino", "f"]), "Gênero"] = 0
    dados.loc[dados["Gênero"].isin(["masculino", "m"]), "Gênero"] = 1
    dados["Gênero"] = pd.to_numeric(dados["Gênero"], errors="coerce").astype("category")

# 2) Rastreamento do gene RET
if "Rastreamento do gene RET" in dados.columns:
    dados["Rastreamento do gene RET"] = dados["Rastreamento do gene RET"].astype(str).str.strip().str.lower()
    dados.loc[dados["Rastreamento do gene RET"].isin(["negativo", "vus", "negativo ou vus"]), "Rastreamento do gene RET"] = 0
    dados.loc[dados["Rastreamento do gene RET"].isin(["risco moderado", "moderado"]), "Rastreamento do gene RET"] = 1
    dados.loc[dados["Rastreamento do gene RET"].isin(["risco alto", "alto"]), "Rastreamento do gene RET"] = 2
    dados.loc[dados["Rastreamento do gene RET"].isin(["risco muito alto", "muito alto", "highest risk"]), "Rastreamento do gene RET"] = 3
    dados["Rastreamento do gene RET"] = pd.to_numeric(dados["Rastreamento do gene RET"], errors="coerce").astype("category")

# 3) Caso índice
if "Probando" in dados.columns:
    dados["Probando"] = dados["Probando"].astype(str).str.strip().str.lower()
    dados.loc[dados["Probando"].isin(["não", "nao", "n", "0"]), "Probando"] = 0
    dados.loc[dados["Probando"].isin(["sim", "s", "1"]), "Probando"] = 1
    dados["Probando"] = pd.to_numeric(dados["Probando"], errors="coerce").astype("category")

# 4) Extensão cirúrgica
if "Extensão cirúrgica" in dados.columns:
    dados["Extensão cirúrgica"] = dados["Extensão cirúrgica"].astype(str).str.strip().str.lower()
    dados.loc[dados["Extensão cirúrgica"].isin(["lobectomia"]), "Extensão cirúrgica"] = 1
    dados.loc[dados["Extensão cirúrgica"].isin(["tireoidectomia total", "total"]), "Extensão cirúrgica"] = 2
    dados.loc[dados["Extensão cirúrgica"].isin(["esvaziamento central", "central"]), "Extensão cirúrgica"] = 3
    dados.loc[dados["Extensão cirúrgica"].isin(["esvaziamento lateral", "lateral"]), "Extensão cirúrgica"] = 4
    dados["Extensão cirúrgica"] = pd.to_numeric(dados["Extensão cirúrgica"], errors="coerce").astype("category")

# 5) Estadiamentos
if "Estadiamento (tamanho)" in dados.columns:
    dados["Estadiamento (tamanho)"] = dados["Estadiamento (tamanho)"].astype("category")

if "Estadiamento (linfonodo)" in dados.columns:
    dados["Estadiamento (linfonodo)"] = dados["Estadiamento (linfonodo)"].astype("category")

# 6) Presença de Carcinoma Diferenciado
if "Presença de Carcinoma Diferenciado" in dados.columns:
    dados["Presença de Carcinoma Diferenciado"] = dados["Presença de Carcinoma Diferenciado"].astype(str).str.strip().str.lower()
    dados.loc[dados["Presença de Carcinoma Diferenciado"].isin(["não", "nao", "n", "0"]), "Presença de Carcinoma Diferenciado"] = 0
    dados.loc[dados["Presença de Carcinoma Diferenciado"].isin(["sim", "s", "1"]), "Presença de Carcinoma Diferenciado"] = 1
    dados["Presença de Carcinoma Diferenciado"] = pd.to_numeric(
        dados["Presença de Carcinoma Diferenciado"], errors="coerce"
    ).astype("category")

# 7) Progressão
if "Progressão" in dados.columns:
    dados["Progressão"] = dados["Progressão"].astype(str).str.strip().str.lower()
    dados.loc[dados["Progressão"].isin(["não", "nao", "n", "0"]), "Progressão"] = 0
    dados.loc[dados["Progressão"].isin(["sim", "s", "1"]), "Progressão"] = 1
    dados["Progressão"] = pd.to_numeric(dados["Progressão"], errors="coerce").astype("category")

#%% Dividindo em X e y

X = dados.drop(columns=["Progressão"], errors="raise").copy()
y = pd.to_numeric(dados["Progressão"].astype(str), errors="coerce")

#%% Descritivo por grupo de Progressão (y) e testes de normalidade e homogeneidade

# Garantir y numérico
y_desc = pd.to_numeric(y.astype(str), errors="coerce")

# Base conjunta
dados_desc = X.copy()
dados_desc["Progressão"] = y_desc
dados_desc = dados_desc.dropna(subset=["Progressão"])

# Identificação de tipos
variaveis_numericas = X.select_dtypes(include=["number"]).columns.tolist()
variaveis_categoricas = X.select_dtypes(include=["category", "object", "bool"]).columns.tolist()

grupos = sorted(dados_desc["Progressão"].dropna().unique())

print("=" * 100)
print("DESCRITIVO - VARIÁVEIS NUMÉRICAS")
print("=" * 100)

# ----------------------------------------
# Numéricas
# ----------------------------------------
if len(variaveis_numericas) > 0 and len(grupos) == 2:

    desc_numerico = dados_desc.groupby("Progressão")[variaveis_numericas].agg(
        ["count", "mean", "std", "median", "min", "max"]
    )

    q1 = dados_desc.groupby("Progressão")[variaveis_numericas].quantile(0.25)
    q3 = dados_desc.groupby("Progressão")[variaveis_numericas].quantile(0.75)

    for var in variaveis_numericas:
        desc_numerico[(var, "p25")] = q1[var]
        desc_numerico[(var, "p75")] = q3[var]

    print(desc_numerico.sort_index(axis=1))

    # ----------------------------------------
    # Testes estatísticos
    # ----------------------------------------
    resultados_testes = []

    g0, g1 = grupos

    for var in variaveis_numericas:
        temp = dados_desc[[var, "Progressão"]].dropna()

        x0 = temp.loc[temp["Progressão"] == g0, var]
        x1 = temp.loc[temp["Progressão"] == g1, var]

        # Shapiro-Wilk
        try:
            p_sw_0 = shapiro(x0)[1] if len(x0) >= 3 else np.nan
        except:
            p_sw_0 = np.nan

        try:
            p_sw_1 = shapiro(x1)[1] if len(x1) >= 3 else np.nan
        except:
            p_sw_1 = np.nan

        # Shapiro-Francia
        try:
            p_sf_0 = shapiro_francia(x0)[1] if len(x0) >= 3 else np.nan
        except:
            p_sf_0 = np.nan

        try:
            p_sf_1 = shapiro_francia(x1)[1] if len(x1) >= 3 else np.nan
        except:
            p_sf_1 = np.nan

        # Levene
        try:
            p_lev = levene(x0, x1, center="median")[1]
        except:
            p_lev = np.nan

        resultados_testes.append({
            "Variável": var,
            f"Shapiro-Wilk p (Y={g0})": p_sw_0,
            f"Shapiro-Wilk p (Y={g1})": p_sw_1,
            f"Shapiro-Francia p (Y={g0})": p_sf_0,
            f"Shapiro-Francia p (Y={g1})": p_sf_1,
            "Levene p": p_lev
        })

    resultados_testes = pd.DataFrame(resultados_testes)

    print("\n" + "=" * 100)
    print("TESTES DE NORMALIDADE E HOMOGENEIDADE")
    print("=" * 100)
    print(resultados_testes)

else:
    print("Não há variáveis numéricas ou y não possui exatamente 2 grupos.")

print("\n" + "=" * 100)
print("DESCRITIVO - VARIÁVEIS CATEGÓRICAS")
print("=" * 100)

# ----------------------------------------
# Categóricas
# ----------------------------------------
if len(variaveis_categoricas) > 0:
    for var in variaveis_categoricas:
        print("\n" + "-" * 80)
        print(f"Variável: {var}")
        print("-" * 80)

        tabela_abs = pd.crosstab(dados_desc[var], dados_desc["Progressão"], dropna=False)
        tabela_perc = pd.crosstab(
            dados_desc[var],
            dados_desc["Progressão"],
            normalize="columns",
            dropna=False
        ) * 100

        print("\nFrequência absoluta:")
        print(tabela_abs)

        print("\nPercentual por coluna (%):")
        print(tabela_perc.round(2))

else:
    print("Não há variáveis categóricas em X.")

#%% Testes para a construção da Tabela 1

def extrair_p_shapiro_francia(x, mostrar=False):
    try:
        res = shapiro_francia(x)

        if mostrar:
            print("Retorno bruto do Shapiro-Francia:")
            print(res)
            print("Tipo:", type(res))
            print("-" * 80)

        # caso 1: tupla/lista
        if isinstance(res, (tuple, list)):
            if len(res) >= 2:
                return float(res[-1])

        # caso 2: pandas Series
        if isinstance(res, pd.Series):
            for chave in ["p-value", "p_value", "pvalor", "p_valor", "p"]:
                if chave in res.index:
                    return float(res[chave])

            # fallback: último valor numérico
            vals = pd.to_numeric(res, errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.iloc[-1])

        # caso 3: pandas DataFrame
        if isinstance(res, pd.DataFrame):
            for chave in ["p-value", "p_value", "pvalor", "p_valor", "p"]:
                if chave in res.columns:
                    return float(res[chave].iloc[0])

            # fallback: último valor numérico da primeira linha
            linha = pd.to_numeric(res.iloc[0], errors="coerce").dropna()
            if len(linha) > 0:
                return float(linha.iloc[-1])

        # caso 4: dict
        if isinstance(res, dict):
            for chave in ["p-value", "p_value", "pvalor", "p_valor", "p"]:
                if chave in res:
                    return float(res[chave])

        return np.nan

    except Exception as e:
        if mostrar:
            print("Erro ao extrair p do Shapiro-Francia:", repr(e))
        return np.nan


dados_teste = X.copy()
dados_teste["y"] = pd.to_numeric(y.astype(str), errors="coerce")
dados_teste = dados_teste.dropna(subset=["y"])

grupos = sorted(dados_teste["y"].dropna().unique())

resultados_numericos = []
resultados_categoricos = []

if len(grupos) != 2:
    print("Erro: y deve ter exatamente 2 grupos.")
else:
    g0, g1 = grupos

    variaveis_numericas = X.select_dtypes(include=["number"]).columns.tolist()
    variaveis_categoricas = X.select_dtypes(include=["category", "object", "bool"]).columns.tolist()

    # -----------------------------
    # Numéricas
    # -----------------------------
    for i, var in enumerate(variaveis_numericas):
        temp = dados_teste[[var, "y"]].dropna()

        x0 = pd.to_numeric(temp.loc[temp["y"] == g0, var], errors="coerce").dropna()
        x1 = pd.to_numeric(temp.loc[temp["y"] == g1, var], errors="coerce").dropna()

        if len(x0) < 3 or len(x1) < 3:
            resultados_numericos.append({
                "Variável": var,
                f"p_SF_Y{g0}": np.nan,
                f"p_SF_Y{g1}": np.nan,
                "p_Levene": np.nan,
                "Normal (ambos grupos)": False,
                "Teste": "Dados insuficientes",
                "Estatística": np.nan,
                "p_valor": np.nan
            })
            continue

        # mostrar retorno bruto só na primeira variável para diagnóstico
        mostrar_debug = (i == 0)

        p_sf0 = extrair_p_shapiro_francia(x0, mostrar=mostrar_debug)
        p_sf1 = extrair_p_shapiro_francia(x1, mostrar=mostrar_debug)

        try:
            p_lev = levene(x0, x1, center="median")[1]
        except:
            p_lev = np.nan

        normal = (
            pd.notna(p_sf0) and
            pd.notna(p_sf1) and
            (p_sf0 > 0.05) and
            (p_sf1 > 0.05)
        )

        if normal:
            if pd.notna(p_lev) and (p_lev > 0.05):
                teste = "Teste t de Student"
                stat, p = ttest_ind(x0, x1, equal_var=True, nan_policy="omit")
            else:
                teste = "Teste t de Welch"
                stat, p = ttest_ind(x0, x1, equal_var=False, nan_policy="omit")
        else:
            teste = "Mann-Whitney"
            stat, p = mannwhitneyu(x0, x1, alternative="two-sided")

        resultados_numericos.append({
            "Variável": var,
            f"p_SF_Y{g0}": p_sf0,
            f"p_SF_Y{g1}": p_sf1,
            "p_Levene": p_lev,
            "Normal (ambos grupos)": normal,
            "Teste": teste,
            "Estatística": stat,
            "p_valor": p
        })

    # -----------------------------
    # Categóricas
    # -----------------------------
    for var in variaveis_categoricas:
        temp = dados_teste[[var, "y"]].dropna()
        tabela = pd.crosstab(temp[var], temp["y"])

        if tabela.shape[0] < 2 or tabela.shape[1] < 2:
            resultados_categoricos.append({
                "Variável": var,
                "Teste": "Tabela insuficiente",
                "Estatística": np.nan,
                "p_valor": np.nan
            })
            continue

        chi2_tab, p_chi, gl, esp = chi2_contingency(tabela)

        if tabela.shape == (2, 2) and (esp < 5).any():
            teste = "Fisher"
            stat, p = fisher_exact(tabela)
        else:
            teste = "Qui-quadrado"
            stat, p = chi2_tab, p_chi

        resultados_categoricos.append({
            "Variável": var,
            "Teste": teste,
            "Estatística": stat,
            "p_valor": p
        })

resultados_numericos = pd.DataFrame(resultados_numericos)
resultados_categoricos = pd.DataFrame(resultados_categoricos)

print("\nNUMÉRICAS")
print(resultados_numericos)

print("\nCATEGÓRICAS")
print(resultados_categoricos)

#%% Tabela 1 - descritivo das variáveis numéricas por grupo de y

variaveis_testadas = []

if not resultados_numericos.empty:
    variaveis_testadas += resultados_numericos["Variável"].dropna().tolist()

if not resultados_categoricos.empty:
    variaveis_testadas += resultados_categoricos["Variável"].dropna().tolist()

variaveis_testadas = list(dict.fromkeys(variaveis_testadas))

print("Variáveis testadas:")
for v in variaveis_testadas:
    print("-", v)

# --------------------------------------------------
# Descritivo das variáveis numéricas conforme o teste
# --------------------------------------------------
resumo_numericas_por_teste = []

# base com y numérico
dados_resumo = X.copy()
dados_resumo["y"] = pd.to_numeric(y.astype(str), errors="coerce")
dados_resumo = dados_resumo.dropna(subset=["y"])

grupos = sorted(dados_resumo["y"].dropna().unique())

if not resultados_numericos.empty and len(grupos) == 2:
    g0, g1 = grupos

    for i in range(len(resultados_numericos)):
        var = resultados_numericos.loc[i, "Variável"]
        teste = resultados_numericos.loc[i, "Teste"]

        if var not in dados_resumo.columns:
            continue

        temp = dados_resumo[[var, "y"]].dropna()

        x0 = pd.to_numeric(temp.loc[temp["y"] == g0, var], errors="coerce").dropna()
        x1 = pd.to_numeric(temp.loc[temp["y"] == g1, var], errors="coerce").dropna()

        if teste in ["Teste t de Student", "Teste t de Welch"]:
            desc_y0 = f"{x0.mean():.2f} ± {x0.std():.2f}"
            desc_y1 = f"{x1.mean():.2f} ± {x1.std():.2f}"

        elif teste == "Mann-Whitney":
            desc_y0 = f"{x0.median():.2f} ({x0.quantile(0.25):.2f}–{x0.quantile(0.75):.2f})"
            desc_y1 = f"{x1.median():.2f} ({x1.quantile(0.25):.2f}–{x1.quantile(0.75):.2f})"

        else:
            desc_y0 = np.nan
            desc_y1 = np.nan

        resumo_numericas_por_teste.append({
            "Variável": var,
            "Teste": teste,
            f"Y={g0}": desc_y0,
            f"Y={g1}": desc_y1
        })

resumo_numericas_por_teste = pd.DataFrame(resumo_numericas_por_teste)

print("\nResumo descritivo das variáveis numéricas por grupo de y:")
print(resumo_numericas_por_teste)

#%% Drop do Tempo de Seguimento

dados = dados.drop(columns=["Tempo de seguimento (anos)"], errors="ignore")
variaveis_testadas = [v for v in variaveis_testadas if v in dados.columns]

#%% Dividindo em X e y novamente (sem o Tempo de Seguimento)

X = dados.drop(columns=["Progressão"], errors="raise").copy()
y = pd.to_numeric(dados["Progressão"].astype(str), errors="coerce")

#%% Funções auxiliares organizadas

def dummizar_variavel(serie, nome_var):
    """
    a) Dummiza uma variável categórica com a seguinte regra:
    - binária: drop_first=True
    - >2 categorias: drop_first=False
    """
    serie = serie.copy()

    if str(serie.dtype) == "category":
        categorias = list(serie.cat.categories)
    else:
        categorias = sorted(serie.dropna().unique().tolist())

    serie = pd.Categorical(serie, categories=categorias, ordered=False)

    n_categorias = len([c for c in categorias if pd.notna(c)])

    if n_categorias <= 1:
        return pd.DataFrame(index=serie.index), {
            "tipo": "categorica",
            "n_categorias": n_categorias,
            "drop_first": None,
            "referencia": np.nan,
            "colunas": []
        }

    # binária -> drop_first=True
    if n_categorias == 2:
        dummies = pd.get_dummies(serie, prefix=nome_var, drop_first=True, dtype=int)
        referencia = categorias[0]
        drop_first = True

    # >2 categorias -> drop_first=False
    else:
        dummies = pd.get_dummies(serie, prefix=nome_var, drop_first=False, dtype=int)
        referencia = "sem referência única no modelo saturado"
        drop_first = False

    meta = {
        "tipo": "categorica",
        "n_categorias": n_categorias,
        "drop_first": drop_first,
        "referencia": referencia,
        "colunas": dummies.columns.tolist()
    }

    return dummies, meta


def preparar_base_modelo(df, y_col, variaveis):
    """
    b) Monta a base para modelagem:
    - numéricas entram como 1 coluna
    - categóricas são dummizadas:
        binárias -> drop_first=True
        >2 categorias -> drop_first=False
    Retorna:
    X_final, y_final, idx_final, mapa_blocos, metadados_blocos
    """
    base = df[variaveis + [y_col]].copy()
    base = base.dropna(subset=[y_col])

    y_base = pd.to_numeric(base[y_col].astype(str), errors="coerce")
    base = base.loc[y_base.notna()].copy()
    y_base = y_base.loc[y_base.notna()].astype(int)

    X_partes = []
    mapa_blocos = {}
    metadados_blocos = {}

    for var in variaveis:
        serie = base[var]

        if str(serie.dtype) in ["category", "object", "bool"]:
            dummies, meta = dummizar_variavel(serie, var)

            if dummies.shape[1] == 0:
                continue

            X_partes.append(dummies)
            mapa_blocos[var] = dummies.columns.tolist()
            metadados_blocos[var] = meta

        else:
            col_num = pd.to_numeric(serie, errors="coerce").to_frame(name=var)
            X_partes.append(col_num)
            mapa_blocos[var] = [var]
            metadados_blocos[var] = {
                "tipo": "numerica",
                "n_categorias": np.nan,
                "drop_first": np.nan,
                "referencia": np.nan,
                "colunas": [var]
            }

    if len(X_partes) == 0:
        raise ValueError("Nenhuma variável elegível para modelagem.")

    X_base = pd.concat(X_partes, axis=1)
    base_modelo = pd.concat([X_base, y_base.rename("y")], axis=1).dropna()

    X_final = base_modelo.drop(columns=["y"]).copy()
    y_final = base_modelo["y"].astype(int).copy()
    idx_final = base_modelo.index.tolist()

    mapa_blocos_final = {}
    metadados_blocos_final = {}

    for var, cols in mapa_blocos.items():
        cols_existentes = [c for c in cols if c in X_final.columns]
        if len(cols_existentes) > 0:
            mapa_blocos_final[var] = cols_existentes
            meta = metadados_blocos[var].copy()
            meta["colunas"] = cols_existentes
            metadados_blocos_final[var] = meta

    return X_final, y_final, idx_final, mapa_blocos_final, metadados_blocos_final


def calcular_metricas_binarias(y_true, prob):
    y_true = pd.Series(y_true).astype(int)
    prob = pd.Series(prob).astype(float)

    auc = roc_auc_score(y_true, prob)

    fpr, tpr, thresholds = roc_curve(y_true, prob)
    youden = tpr - fpr
    idx = np.argmax(youden)
    corte = thresholds[idx]

    pred = (prob >= corte).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    esp = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    vpp = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    vpn = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    f1 = 2 * (vpp * sens) / (vpp + sens) if pd.notna(vpp) and pd.notna(sens) and (vpp + sens) > 0 else np.nan

    metricas = {
        "AUC": auc,
        "Ponto_Youden": corte,
        "Sensibilidade": sens,
        "Especificidade": esp,
        "VPP": vpp,
        "VPN": vpn,
        "Acuracia": acc,
        "F1_Score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    return metricas, pred


def hosmer_lemeshow_test(y_true, y_pred, g=10):
    y_true = pd.Series(y_true).astype(float).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(float).reset_index(drop=True)

    dados_hl = pd.DataFrame({"y": y_true, "p": y_pred})
    dados_hl["p"] = np.clip(dados_hl["p"], 1e-8, 1 - 1e-8)

    try:
        dados_hl["grupo"] = pd.qcut(dados_hl["p"], q=g, duplicates="drop")
    except ValueError:
        n_grupos_validos = min(g, dados_hl["p"].nunique())
        dados_hl["grupo"] = pd.qcut(dados_hl["p"], q=n_grupos_validos, duplicates="drop")

    tabela_hl = dados_hl.groupby("grupo", observed=False).agg(
        n=("y", "count"),
        eventos_observados=("y", "sum"),
        eventos_esperados=("p", "sum")
    )

    tabela_hl["nao_eventos_observados"] = tabela_hl["n"] - tabela_hl["eventos_observados"]
    tabela_hl["nao_eventos_esperados"] = tabela_hl["n"] - tabela_hl["eventos_esperados"]

    eps = 1e-8
    tabela_hl["eventos_esperados"] = np.clip(tabela_hl["eventos_esperados"], eps, None)
    tabela_hl["nao_eventos_esperados"] = np.clip(tabela_hl["nao_eventos_esperados"], eps, None)

    hl_stat = (
        ((tabela_hl["eventos_observados"] - tabela_hl["eventos_esperados"]) ** 2) / tabela_hl["eventos_esperados"]
        + ((tabela_hl["nao_eventos_observados"] - tabela_hl["nao_eventos_esperados"]) ** 2) / tabela_hl["nao_eventos_esperados"]
    ).sum()

    gl = max(len(tabela_hl) - 2, 1)
    p_value = 1 - chi2.cdf(hl_stat, gl)

    return hl_stat, gl, p_value, tabela_hl.reset_index()


def pseudo_r2_logit(modelo_logit, y):
    ll_modelo = modelo_logit.llf
    y = pd.Series(y).astype(int)
    p_nulo = y.mean()

    ll_nulo = np.sum(
        y * np.log(np.clip(p_nulo, 1e-8, 1 - 1e-8)) +
        (1 - y) * np.log(np.clip(1 - p_nulo, 1e-8, 1 - 1e-8))
    )

    n = len(y)

    r2_cox_snell = 1 - np.exp((2 / n) * (ll_nulo - ll_modelo))
    r2_nagelkerke = r2_cox_snell / (1 - np.exp((2 / n) * ll_nulo))

    return r2_cox_snell, r2_nagelkerke


def stepwise_logistico_por_blocos(X, y, mapa_blocos, p_enter=0.05, p_remove=0.10, verbose=True):
    """
    c) Stepwise por blocos.
    Cada variável categórica entra como bloco de dummies.
    """
    blocos_disponiveis = list(mapa_blocos.keys())
    blocos_selecionados = []
    mudou = True

    while mudou:
        mudou = False

        # forward
        pvals_forward = pd.Series(dtype=float)

        for bloco in blocos_disponiveis:
            if bloco in blocos_selecionados:
                continue

            try:
                cols_reduzido = []
                for b in blocos_selecionados:
                    cols_reduzido += mapa_blocos[b]

                cols_completo = cols_reduzido + mapa_blocos[bloco]

                X_red = sm.add_constant(X[cols_reduzido], has_constant="add") if len(cols_reduzido) > 0 else sm.add_constant(pd.DataFrame(index=X.index), has_constant="add")
                X_com = sm.add_constant(X[cols_completo], has_constant="add")

                m_red = sm.Logit(y, X_red).fit(disp=0)
                m_com = sm.Logit(y, X_com).fit(disp=0)

                lr = 2 * (m_com.llf - m_red.llf)
                gl = len(mapa_blocos[bloco])
                p_lr = 1 - chi2.cdf(lr, gl)

                pvals_forward.loc[bloco] = p_lr

            except:
                pvals_forward.loc[bloco] = np.nan

        if not pvals_forward.empty:
            melhor_p = pvals_forward.min()
            if pd.notna(melhor_p) and melhor_p < p_enter:
                melhor_bloco = pvals_forward.idxmin()
                blocos_selecionados.append(melhor_bloco)
                mudou = True
                if verbose:
                    print("Entrou bloco:", melhor_bloco, "| p =", melhor_p)

        # backward
        if len(blocos_selecionados) > 0:
            pvals_backward = pd.Series(dtype=float)

            for bloco in blocos_selecionados:
                try:
                    cols_full = []
                    for b in blocos_selecionados:
                        cols_full += mapa_blocos[b]

                    cols_red = []
                    for b in blocos_selecionados:
                        if b != bloco:
                            cols_red += mapa_blocos[b]

                    X_full = sm.add_constant(X[cols_full], has_constant="add")
                    X_red = sm.add_constant(X[cols_red], has_constant="add") if len(cols_red) > 0 else sm.add_constant(pd.DataFrame(index=X.index), has_constant="add")

                    m_full = sm.Logit(y, X_full).fit(disp=0)
                    m_red = sm.Logit(y, X_red).fit(disp=0)

                    lr = 2 * (m_full.llf - m_red.llf)
                    gl = len(mapa_blocos[bloco])
                    p_lr = 1 - chi2.cdf(lr, gl)

                    pvals_backward.loc[bloco] = p_lr

                except:
                    pvals_backward.loc[bloco] = np.nan

            pior_p = pvals_backward.max()
            if pd.notna(pior_p) and pior_p > p_remove:
                pior_bloco = pvals_backward.idxmax()
                blocos_selecionados.remove(pior_bloco)
                mudou = True
                if verbose:
                    print("Saiu bloco:", pior_bloco, "| p =", pior_p)

    return blocos_selecionados

#%% Função auxiliar para rotular variáveis dummizadas nos gráficos e tabelas

def rotular_coluna_dummy(nome_coluna, colunas_originais):
    """
    Converte nomes de colunas dummizadas em rótulos legíveis.
    Exemplo:
    - 'Extensão cirúrgica_4' -> 'Extensão cirúrgica = 4'
    - 'Gênero_1' -> 'Gênero = 1'
    - 'Idade' -> 'Idade'
    """
    if nome_coluna in colunas_originais:
        return nome_coluna

    correspondencias = [
        col for col in colunas_originais
        if nome_coluna.startswith(f"{col}_")
    ]

    if len(correspondencias) == 0:
        return nome_coluna

    # usa a correspondência mais longa para evitar ambiguidade
    base = sorted(correspondencias, key=len, reverse=True)[0]
    categoria = nome_coluna[len(base) + 1:]

    return f"{base} = {categoria}"

#%% Avaliação diagnóstica das variáveis categóricas

desfecho = "Progressão"
limite_categoria_rara = 5

dados_diag = dados.copy()
dados_diag[desfecho] = pd.to_numeric(dados_diag[desfecho].astype(str), errors="coerce")
dados_diag = dados_diag.dropna(subset=[desfecho])
dados_diag[desfecho] = dados_diag[desfecho].astype(int)

variaveis_categoricas_diag = X.select_dtypes(include=["category", "object", "bool"]).columns.tolist()

print("=" * 100)
print("VARIÁVEIS CATEGÓRICAS IDENTIFICADAS")
print("=" * 100)
for v in variaveis_categoricas_diag:
    print("-", v)

resumo_diagnostico = []

for var in variaveis_categoricas_diag:
    base = dados_diag[[var, desfecho]].copy().dropna()

    print("\n" + "=" * 100)
    print(f"VARIÁVEL: {var}")
    print("=" * 100)

    if base.empty:
        print("Sem dados após remoção de missing.")
        resumo_diagnostico.append({
            "Variável": var,
            "n_total": 0,
            "n_categorias": 0,
            "categoria_rara": "Sim",
            "celula_zero": "Não avaliado",
            "separacao_completa": "Não avaliado",
            "esperado_<5_em_2x2": "Não avaliado",
            "status_risco": "ALTO"
        })
        continue

    freq_abs = base[var].value_counts(dropna=False).sort_index()
    freq_rel = base[var].value_counts(dropna=False, normalize=True).sort_index() * 100

    tabela_freq = pd.DataFrame({
        "n": freq_abs,
        "%": freq_rel.round(2)
    })

    print("\nFrequências:")
    print(tabela_freq)

    tabela_cont = pd.crosstab(base[var], base[desfecho], dropna=False)

    print("\nTabela de contingência vs Progressão:")
    print(tabela_cont)

    categoria_rara = (freq_abs < limite_categoria_rara).any()
    celula_zero = (tabela_cont == 0).any().any()

    separacao_completa = False
    for idx in tabela_cont.index:
        linha = tabela_cont.loc[idx]
        if (linha == 0).any():
            separacao_completa = True

    esperado_menor_5 = False

    try:
        chi2_stat, p_chi, gl, esperados = chi2_contingency(tabela_cont)
        tabela_esperada_df = pd.DataFrame(
            esperados,
            index=tabela_cont.index,
            columns=tabela_cont.columns
        )

        print("\nValores esperados:")
        print(tabela_esperada_df.round(3))

        if tabela_cont.shape == (2, 2):
            esperado_menor_5 = (esperados < 5).any()

    except Exception as e:
        print("\nNão foi possível calcular os valores esperados.")
        print("Erro:", repr(e))

    fatores_risco = []

    if categoria_rara:
        fatores_risco.append("categoria rara")
    if celula_zero:
        fatores_risco.append("célula zero")
    if separacao_completa:
        fatores_risco.append("separação completa")
    if esperado_menor_5:
        fatores_risco.append("esperado < 5 em 2x2")

    if len(fatores_risco) == 0:
        status_risco = "BAIXO"
    elif len(fatores_risco) == 1:
        status_risco = "MODERADO"
    else:
        status_risco = "ALTO"

    print("\nDiagnóstico:")
    print("Categoria rara:", "Sim" if categoria_rara else "Não")
    print("Célula zero:", "Sim" if celula_zero else "Não")
    print("Separação completa:", "Sim" if separacao_completa else "Não")
    print("Esperado < 5 em tabela 2x2:", "Sim" if esperado_menor_5 else "Não")
    print("Fatores de risco:", ", ".join(fatores_risco) if fatores_risco else "Nenhum")
    print("Status final:", status_risco)

    resumo_diagnostico.append({
        "Variável": var,
        "n_total": len(base),
        "n_categorias": base[var].nunique(dropna=True),
        "categoria_rara": "Sim" if categoria_rara else "Não",
        "celula_zero": "Sim" if celula_zero else "Não",
        "separacao_completa": "Sim" if separacao_completa else "Não",
        "esperado_<5_em_2x2": "Sim" if esperado_menor_5 else "Não",
        "fatores_risco": ", ".join(fatores_risco) if fatores_risco else "Nenhum",
        "status_risco": status_risco
    })

resumo_diagnostico = pd.DataFrame(resumo_diagnostico)

print("\n" + "=" * 100)
print("RESUMO DIAGNÓSTICO DAS VARIÁVEIS CATEGÓRICAS")
print("=" * 100)
print(resumo_diagnostico.sort_values(by=["status_risco", "Variável"], ascending=[False, True]))

#%% I) Regressão logística univariada de cada variável estudada

def regressao_logistica_univariada_bloco(df, y_col, variaveis):
    resultados = []
    probs_univariadas = pd.DataFrame(index=df.index)

    for var in variaveis:
        base = df[[var, y_col]].copy().dropna()
        if base.empty:
            continue

        y_var = pd.to_numeric(base[y_col].astype(str), errors="coerce")
        base = base.loc[y_var.notna()].copy()
        y_var = y_var.loc[y_var.notna()].astype(int)

        serie = base[var]

        # Variável categórica
        if str(serie.dtype) in ["category", "object", "bool"]:
            X_var = pd.get_dummies(serie, prefix=var, drop_first=True, dtype=int)

            if X_var.shape[1] == 0:
                continue

            try:
                X_sm = sm.add_constant(X_var, has_constant="add")
                modelo = sm.Logit(y_var, X_sm).fit(disp=0)

                prob = modelo.predict(X_sm)
                metricas, _ = calcular_metricas_binarias(y_var, prob)

                nome_prob = f"proba_uni_{var}"
                probs_univariadas.loc[base.index, nome_prob] = prob

                conf = modelo.conf_int()

                # teste global aproximado
                X0 = sm.add_constant(pd.DataFrame(index=X_var.index), has_constant="add")
                m0 = sm.Logit(y_var, X0).fit(disp=0)
                lr_stat = 2 * (modelo.llf - m0.llf)
                gl = X_var.shape[1]
                p_global = 1 - chi2.cdf(lr_stat, gl)

                for termo in modelo.params.index:
                    if termo == "const":
                        continue

                    resultados.append({
                        "Variável": var,
                        "Tipo": "Categórica",
                        "Termo": termo,
                        "Coeficiente": modelo.params[termo],
                        "OR": np.exp(modelo.params[termo]),
                        "IC95_inf": np.exp(conf.loc[termo, 0]),
                        "IC95_sup": np.exp(conf.loc[termo, 1]),
                        "p_valor_termo": modelo.pvalues[termo],
                        "LR_global": lr_stat,
                        "gl_global": gl,
                        "p_valor_global": p_global,
                        "AUC": metricas["AUC"],
                        "Ponto_Youden": metricas["Ponto_Youden"],
                        "Sensibilidade": metricas["Sensibilidade"],
                        "Especificidade": metricas["Especificidade"],
                        "VPP": metricas["VPP"],
                        "VPN": metricas["VPN"],
                        "F1_Score": metricas["F1_Score"],
                        "Acuracia": metricas["Acuracia"],
                        "n": int(modelo.nobs)
                    })

            except Exception as e:
                resultados.append({
                    "Variável": var,
                    "Tipo": "Categórica",
                    "Termo": f"falha_ajuste: {repr(e)}",
                    "Coeficiente": np.nan,
                    "OR": np.nan,
                    "IC95_inf": np.nan,
                    "IC95_sup": np.nan,
                    "p_valor_termo": np.nan,
                    "LR_global": np.nan,
                    "gl_global": np.nan,
                    "p_valor_global": np.nan,
                    "AUC": np.nan,
                    "Ponto_Youden": np.nan,
                    "Sensibilidade": np.nan,
                    "Especificidade": np.nan,
                    "VPP": np.nan,
                    "VPN": np.nan,
                    "F1_Score": np.nan,
                    "Acuracia": np.nan,
                    "n": np.nan
                })

        # Variável numérica
        else:
            x_num = pd.to_numeric(serie, errors="coerce")
            base2 = pd.concat([x_num.rename(var), y_var.rename("y")], axis=1).dropna()

            if base2.empty:
                continue

            X_num = base2[[var]]
            y_num = base2["y"].astype(int)

            try:
                X_sm = sm.add_constant(X_num, has_constant="add")
                modelo = sm.Logit(y_num, X_sm).fit(disp=0)

                prob = modelo.predict(X_sm)
                metricas, _ = calcular_metricas_binarias(y_num, prob)

                nome_prob = f"proba_uni_{var}"
                probs_univariadas.loc[base2.index, nome_prob] = prob

                conf = modelo.conf_int()

                resultados.append({
                    "Variável": var,
                    "Tipo": "Numérica",
                    "Termo": var,
                    "Coeficiente": modelo.params[var],
                    "OR": np.exp(modelo.params[var]),
                    "IC95_inf": np.exp(conf.loc[var, 0]),
                    "IC95_sup": np.exp(conf.loc[var, 1]),
                    "p_valor_termo": modelo.pvalues[var],
                    "LR_global": np.nan,
                    "gl_global": 1,
                    "p_valor_global": modelo.pvalues[var],
                    "AUC": metricas["AUC"],
                    "Ponto_Youden": metricas["Ponto_Youden"],
                    "Sensibilidade": metricas["Sensibilidade"],
                    "Especificidade": metricas["Especificidade"],
                    "VPP": metricas["VPP"],
                    "VPN": metricas["VPN"],
                    "F1_Score": metricas["F1_Score"],
                    "Acuracia": metricas["Acuracia"],
                    "n": int(modelo.nobs)
                })

            except Exception as e:
                resultados.append({
                    "Variável": var,
                    "Tipo": "Numérica",
                    "Termo": f"falha_ajuste: {repr(e)}",
                    "Coeficiente": np.nan,
                    "OR": np.nan,
                    "IC95_inf": np.nan,
                    "IC95_sup": np.nan,
                    "p_valor_termo": np.nan,
                    "LR_global": np.nan,
                    "gl_global": np.nan,
                    "p_valor_global": np.nan,
                    "AUC": np.nan,
                    "Ponto_Youden": np.nan,
                    "Sensibilidade": np.nan,
                    "Especificidade": np.nan,
                    "VPP": np.nan,
                    "VPN": np.nan,
                    "Acuracia": np.nan,
                    "n": np.nan
                })

    return pd.DataFrame(resultados), probs_univariadas


resultados_logit_uni, probs_univariadas = regressao_logistica_univariada_bloco(
    dados, "Progressão", variaveis_testadas
)

print("=" * 100)
print("REGRESSÃO LOGÍSTICA UNIVARIADA")
print("=" * 100)
print(resultados_logit_uni)

#%% Base multivariada organizada

X_modelo, y_modelo, idx_modelo, mapa_blocos, metadados_blocos = preparar_base_modelo(
    dados, "Progressão", variaveis_testadas
)

print("Dimensão da base modelada:", X_modelo.shape)
print("Eventos:", y_modelo.sum(), "| Não eventos:", (y_modelo == 0).sum())

print("\nBlocos de variáveis:")
for bloco, cols in mapa_blocos.items():
    meta = metadados_blocos[bloco]
    print(
        f"- {bloco}: {cols} | "
        f"tipo={meta['tipo']} | "
        f"n_categorias={meta['n_categorias']} | "
        f"drop_first={meta['drop_first']} | "
        f"referência={meta['referencia']}"
    )

#%% Regressão Logística Multivariada (Stepwise - statstests)

# --------------------------------------------------
# Base para fórmula com nomes seguros
# --------------------------------------------------
df_modelo_formula = X_modelo.copy()
df_modelo_formula["y"] = y_modelo.values

mapeamento_colunas_seguras = {
    col: f"var_{i}"
    for i, col in enumerate(X_modelo.columns, start=1)
}

mapeamento_reverso = {v: k for k, v in mapeamento_colunas_seguras.items()}

df_modelo_formula = df_modelo_formula.rename(columns=mapeamento_colunas_seguras)

formula_completa = "y ~ " + " + ".join(mapeamento_colunas_seguras.values())

print("Fórmula inicial completa:")
print(formula_completa)

# --------------------------------------------------
# Modelo inicial
# --------------------------------------------------
modelo_inicial_stepwise = sm.Logit.from_formula(
    formula_completa,
    data=df_modelo_formula
).fit(disp=0)

# --------------------------------------------------
# Stepwise do statstests
# --------------------------------------------------
modelo_logit_multi = stepwise(
    modelo_inicial_stepwise,
    pvalue_limit=0.05
)

print("\nResumo do modelo final após stepwise:")
print(modelo_logit_multi.summary())

# --------------------------------------------------
# Variáveis selecionadas
# --------------------------------------------------
variaveis_stepwise_seguras = [
    v for v in modelo_logit_multi.params.index
    if v != "Intercept"
]

variaveis_stepwise = [
    mapeamento_reverso.get(v, v)
    for v in variaveis_stepwise_seguras
]

print("\nVariáveis selecionadas no stepwise:")
for v in variaveis_stepwise:
    print("-", v)

if len(variaveis_stepwise_seguras) == 0:
    raise ValueError("Nenhuma variável foi selecionada pelo stepwise.")

# --------------------------------------------------
# Predições do modelo final
# --------------------------------------------------
proba_logit_multi = modelo_logit_multi.predict(df_modelo_formula)
metricas_logit_multi, pred_logit_multi = calcular_metricas_binarias(y_modelo, proba_logit_multi)

# --------------------------------------------------
# Hosmer-Lemeshow e pseudo-R²
# --------------------------------------------------
hl_stat, hl_gl, hl_p, tabela_hl_multi = hosmer_lemeshow_test(y_modelo, proba_logit_multi, g=10)
r2_cs_multi, r2_nk_multi = pseudo_r2_logit(modelo_logit_multi, y_modelo)

# --------------------------------------------------
# Tabela de coeficientes do modelo final
# --------------------------------------------------
conf_multi = modelo_logit_multi.conf_int()

resultado_logit_multi = pd.DataFrame({
    "Termo": [
        "Intercepto" if termo == "Intercept" else mapeamento_reverso.get(termo, termo)
        for termo in modelo_logit_multi.params.index
    ],
    "Coeficiente": modelo_logit_multi.params.values,
    "OR": np.exp(modelo_logit_multi.params.values),
    "IC95_inf": np.exp(conf_multi[0].values),
    "IC95_sup": np.exp(conf_multi[1].values),
    "p_valor": modelo_logit_multi.pvalues.values
})

# --------------------------------------------------
# Teste global por variável (LR: modelo completo vs reduzido)
# --------------------------------------------------
resultados_variaveis_multi = []

for var_segura in variaveis_stepwise_seguras:
    try:
        vars_reduzidas = [v for v in variaveis_stepwise_seguras if v != var_segura]

        formula_full = "y ~ " + " + ".join(variaveis_stepwise_seguras)

        if len(vars_reduzidas) > 0:
            formula_red = "y ~ " + " + ".join(vars_reduzidas)
        else:
            formula_red = "y ~ 1"

        m_full = sm.Logit.from_formula(formula_full, data=df_modelo_formula).fit(disp=0)
        m_red = sm.Logit.from_formula(formula_red, data=df_modelo_formula).fit(disp=0)

        lr = 2 * (m_full.llf - m_red.llf)
        gl = 1
        p_lr = 1 - chi2.cdf(lr, gl)

    except Exception:
        lr, gl, p_lr = np.nan, np.nan, np.nan

    resultados_variaveis_multi.append({
        "Variavel": mapeamento_reverso.get(var_segura, var_segura),
        "LR_global": lr,
        "gl_global": gl,
        "p_valor_global": p_lr
    })

resultados_variaveis_multi = pd.DataFrame(resultados_variaveis_multi)

# --------------------------------------------------
# Qualidade do ajuste
# --------------------------------------------------
qualidade_logit_multi = pd.DataFrame([{
    "Hosmer_Lemeshow": hl_stat,
    "gl_HL": hl_gl,
    "p_HL": hl_p,
    "Cox_Snell_R2": r2_cs_multi,
    "Nagelkerke_R2": r2_nk_multi,
    "AUC": metricas_logit_multi["AUC"],
    "Ponto_Youden": metricas_logit_multi["Ponto_Youden"],
    "Sensibilidade": metricas_logit_multi["Sensibilidade"],
    "Especificidade": metricas_logit_multi["Especificidade"],
    "VPP": metricas_logit_multi["VPP"],
    "VPN": metricas_logit_multi["VPN"],
    "Acuracia": metricas_logit_multi["Acuracia"],
    "F1_Score": metricas_logit_multi["F1_Score"]
}])

# --------------------------------------------------
# Saídas
# --------------------------------------------------
print("\nCoeficientes do modelo final")
print(resultado_logit_multi)

print("\nContribuição global das variáveis no modelo final")
print(resultados_variaveis_multi)

print("\nQualidade do ajuste - modelo multivariado")
print(qualidade_logit_multi)

print("\nTabela do Hosmer-Lemeshow")
print(tabela_hl_multi)

#%% II) Árvore de Decisão - processamento inicial dos dados

# Matriz numérica para o sklearn (transformando categóricas em dummies)
X_tree = pd.get_dummies(X, drop_first=True, dtype=int)

# y como binário inteiro
y_tree = pd.to_numeric(y, errors="coerce").astype(int)

# Rótulos legíveis das variáveis do modelo
rotulos_legiveis_modelos = {
    col: rotular_coluna_dummy(col, X.columns.tolist())
    for col in X_tree.columns.tolist()
}

print("Dimensão de X_tree:", X_tree.shape)
print("Variáveis preditoras usadas na árvore:")
print(X_tree.columns.tolist())

#%% Separando as amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    X_tree,
    y_tree,
    test_size=0.30,
    random_state=100,
    stratify=y_tree
)

print("Dimensão treino:", X_train.shape)
print("Dimensão teste:", X_test.shape)

#%% GridSearch para Árvore de Decisão

param_grid_tree = {
    "max_depth": [3, 4, 5, 6, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 3, 5, 10],
    "class_weight": ["balanced", None],
    "criterion": ["gini", "entropy", "log_loss"]
}

tree_grid = DecisionTreeClassifier(random_state=100)

grid_tree = GridSearchCV(
    estimator=tree_grid,
    param_grid=param_grid_tree,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_tree.fit(X_train, y_train)

print("Melhores parâmetros da árvore:")
print(grid_tree.best_params_)

print("\nMelhor AUC média na validação cruzada:")
print(grid_tree.best_score_)

# Melhor modelo
tree_best = grid_tree.best_estimator_

# Predições
tree_best_pred_train = tree_best.predict(X_train)
tree_best_prob_train = tree_best.predict_proba(X_train)[:, 1]

tree_best_pred_test = tree_best.predict(X_test)
tree_best_prob_test = tree_best.predict_proba(X_test)[:, 1]

# Métricas pelo seu pipeline
metricas_tree_grid_train, pred_tree_grid_train = calcular_metricas_binarias(y_train, tree_best_prob_train)
metricas_tree_grid_test, pred_tree_grid_test = calcular_metricas_binarias(y_test, tree_best_prob_test)

print("\nMétricas da Árvore - treino")
print(metricas_tree_grid_train)

print("\nMétricas da Árvore - teste")
print(metricas_tree_grid_test)

tree_pred_test_prob = tree_best.predict_proba(X_test)

#%% Gerando a árvore de decisão inicial

tree_clf = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=15,
    class_weight='balanced',
    random_state=100
)

tree_clf.fit(X_train, y_train)

#%% Plotando a árvore inicial

plt.figure(figsize=(16, 8), dpi=600)

plot_tree(
    tree_clf,
    feature_names=X_tree.columns.tolist(),
    class_names=['Sem Progressão', 'Com Progressão'],
    proportion=False,
    filled=True,
    node_ids=True,
    rounded=True,          
    fontsize=16,           
    impurity=True,
    label='all'
)

plt.tight_layout(pad=1.5)
plt.show()

#%% Importância das variáveis preditoras - árvore inicial

import matplotlib.pyplot as plt

tree_features = pd.DataFrame({
    "features": [rotulos_legiveis_modelos[c] for c in X_tree.columns.tolist()],
    "importance": tree_clf.feature_importances_
}).sort_values(by="importance", ascending=False)

print(tree_features)

# --------------------------------------------------
# Gráfico de importância das variáveis
# --------------------------------------------------

top_n = 10  # você pode ajustar (ex: 5, 10, todas)
tree_features_plot = tree_features.head(top_n)

plt.figure(figsize=(10, 6), dpi=150)

plt.barh(
    tree_features_plot["features"][::-1],
    tree_features_plot["importance"][::-1]
)

plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.title("Importância das Variáveis - Árvore de Decisão")

plt.tight_layout()
plt.show()

#%% III) Random Forest - Estimando uma Random Forest

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    max_features="sqrt",
    random_state=100
)
rf_clf.fit(X_train, y_train)

#%% Grid Search para Random Forest

param_grid_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 4, 5, 6, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 3, 5, 10]
}

rf_grid = RandomForestClassifier(random_state=100)

rf_grid_model = GridSearchCV(
    estimator=rf_grid,
    param_grid=param_grid_rf,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

rf_grid_model.fit(X_train, y_train)

print("Melhores parâmetros da Random Forest:")
print(rf_grid_model.best_params_)

print("\nMelhor AUC média na validação cruzada:")
print(rf_grid_model.best_score_)

rf_best = rf_grid_model.best_estimator_

#%% Obtendo os valores preditos pela Random Forest

rf_pred_train_class = rf_best.predict(X_train)
rf_pred_train_prob = rf_best.predict_proba(X_train)

rf_pred_test_class = rf_best.predict(X_test)
rf_pred_test_prob = rf_best.predict_proba(X_test)

#%% Matriz de confusão e métricas (base de treino)

# Probabilidades e classes preditas com ponto de corte de Youden
metricas_rf_train, rf_pred_train_class = calcular_metricas_binarias(y_train, rf_pred_train_prob[:, 1])

# Matriz de confusão
rf_cm_train = confusion_matrix(y_train, rf_pred_train_class)
cm_rf_train = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm_train,
    display_labels=['Sem Progressão', 'Com Progressão']
)

plt.figure(figsize=(8, 6), dpi=600)
cm_rf_train.plot(colorbar=False, cmap='Oranges', values_format='d')
plt.title('Random Forest: Treino')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.tight_layout()
plt.show()

# Tabela de métricas
metricas_rf_train_df = pd.DataFrame([{
    "Base": "Treino",
    "AUC": metricas_rf_train["AUC"],
    "Ponto_Youden": metricas_rf_train["Ponto_Youden"],
    "Sensibilidade": metricas_rf_train["Sensibilidade"],
    "Especificidade": metricas_rf_train["Especificidade"],
    "VPP": metricas_rf_train["VPP"],
    "VPN": metricas_rf_train["VPN"],
    "Acuracia": metricas_rf_train["Acuracia"],
    "F1_Score": metricas_rf_train["F1_Score"],
    "TN": metricas_rf_train["TN"],
    "FP": metricas_rf_train["FP"],
    "FN": metricas_rf_train["FN"],
    "TP": metricas_rf_train["TP"]
}])

print("Avaliação da Random Forest (Base de Treino)")
print(metricas_rf_train_df)

print(f"\nAUC: {metricas_rf_train['AUC']:.3f}")
print(f"Ponto de corte (Youden): {metricas_rf_train['Ponto_Youden']:.3f}")
print(f"Sensibilidade: {metricas_rf_train['Sensibilidade']:.1%}")
print(f"Especificidade: {metricas_rf_train['Especificidade']:.1%}")
print(f"VPP: {metricas_rf_train['VPP']:.1%}")
print(f"VPN: {metricas_rf_train['VPN']:.1%}")
print(f"Acurácia: {metricas_rf_train['Acuracia']:.1%}")
print(f"F1 Score: {metricas_rf_train['F1_Score']:.3f}")
print(f"TN: {metricas_rf_train['TN']}")
print(f"FP: {metricas_rf_train['FP']}")
print(f"FN: {metricas_rf_train['FN']}")
print(f"TP: {metricas_rf_train['TP']}")

#%% Matriz de confusão e métricas (base de teste)

# Probabilidades e classes preditas com ponto de corte de Youden
metricas_rf_test, rf_pred_test_class = calcular_metricas_binarias(y_test, rf_pred_test_prob[:, 1])

# Matriz de confusão
rf_cm_test = confusion_matrix(y_test, rf_pred_test_class)
cm_rf_test = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm_test,
    display_labels=['Sem Progressão', 'Com Progressão']
)

plt.figure(figsize=(8, 6), dpi=600)
cm_rf_test.plot(colorbar=False, cmap='Oranges', values_format='d')
plt.title('Random Forest: Teste')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.tight_layout()
plt.show()

# Tabela de métricas
metricas_rf_test_df = pd.DataFrame([{
    "Base": "Teste",
    "AUC": metricas_rf_test["AUC"],
    "Ponto_Youden": metricas_rf_test["Ponto_Youden"],
    "Sensibilidade": metricas_rf_test["Sensibilidade"],
    "Especificidade": metricas_rf_test["Especificidade"],
    "VPP": metricas_rf_test["VPP"],
    "VPN": metricas_rf_test["VPN"],
    "Acuracia": metricas_rf_test["Acuracia"],
    "F1_Score": metricas_rf_test["F1_Score"],
    "TN": metricas_rf_test["TN"],
    "FP": metricas_rf_test["FP"],
    "FN": metricas_rf_test["FN"],
    "TP": metricas_rf_test["TP"]
}])

print("Avaliação da Random Forest (Base de Teste)")
print(metricas_rf_test_df)

print(f"\nAUC: {metricas_rf_test['AUC']:.3f}")
print(f"Ponto de corte (Youden): {metricas_rf_test['Ponto_Youden']:.3f}")
print(f"Sensibilidade: {metricas_rf_test['Sensibilidade']:.1%}")
print(f"Especificidade: {metricas_rf_test['Especificidade']:.1%}")
print(f"VPP: {metricas_rf_test['VPP']:.1%}")
print(f"VPN: {metricas_rf_test['VPN']:.1%}")
print(f"Acurácia: {metricas_rf_test['Acuracia']:.1%}")
print(f"F1 Score: {metricas_rf_test['F1_Score']:.3f}")
print(f"TN: {metricas_rf_test['TN']}")
print(f"FP: {metricas_rf_test['FP']}")
print(f"FN: {metricas_rf_test['FN']}")
print(f"TP: {metricas_rf_test['TP']}")

#%% Importância das variáveis preditoras

rf_features = pd.DataFrame({
    'features': [rotulos_legiveis_modelos[c] for c in X_tree.columns.tolist()],
    'importance': rf_best.feature_importances_
}).sort_values(by='importance', ascending=False)

print(rf_features)

# --------------------------------------------------
# Gráfico de importância das variáveis
# --------------------------------------------------
top_n = 10
rf_features_plot = rf_features.head(top_n)

plt.figure(figsize=(10, 6), dpi=600)
plt.barh(
    rf_features_plot["features"][::-1],
    rf_features_plot["importance"][::-1],
    color='darkorange'
)
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.title("Importância das Variáveis - Random Forest")
plt.tight_layout()
plt.show()

#%% IV) LightGBM - Estimando um LightGBM

lgbm_clf = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=100
)
lgbm_clf.fit(X_train, y_train)

#%% Grid Search para LightGBM

param_grid_lgbm = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'max_depth': [3, 4, 5, 6, None],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

lgbm_grid = LGBMClassifier(
    objective='binary',
    random_state=100,
    verbosity=-1
)

lgbm_grid_model = GridSearchCV(
    estimator=lgbm_grid,
    param_grid=param_grid_lgbm,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

lgbm_grid_model.fit(X_train, y_train)

print("Melhores parâmetros do LightGBM:")
print(lgbm_grid_model.best_params_)

print("\nMelhor AUC média na validação cruzada:")
print(lgbm_grid_model.best_score_)

lgbm_best = lgbm_grid_model.best_estimator_

#%% Obtendo os valores preditos pelo LightGBM

lgbm_pred_train_class = lgbm_best.predict(X_train)
lgbm_pred_train_prob = lgbm_best.predict_proba(X_train)

lgbm_pred_test_class = lgbm_best.predict(X_test)
lgbm_pred_test_prob = lgbm_best.predict_proba(X_test)

#%% Matriz de confusão e métricas (base de treino)

# Probabilidades e classes preditas com ponto de corte de Youden
metricas_lgbm_train, lgbm_pred_train_class = calcular_metricas_binarias(y_train, lgbm_pred_train_prob[:, 1])

# Matriz de confusão
lgbm_cm_train = confusion_matrix(y_train, lgbm_pred_train_class)
cm_lgbm_train = ConfusionMatrixDisplay(
    confusion_matrix=lgbm_cm_train,
    display_labels=['Sem Progressão', 'Com Progressão']
)

plt.figure(figsize=(8, 6), dpi=600)
cm_lgbm_train.plot(colorbar=False, cmap='Purples', values_format='d')
plt.title('LightGBM: Treino')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.tight_layout()
plt.show()

# Tabela de métricas
metricas_lgbm_train_df = pd.DataFrame([{
    "Base": "Treino",
    "AUC": metricas_lgbm_train["AUC"],
    "Ponto_Youden": metricas_lgbm_train["Ponto_Youden"],
    "Sensibilidade": metricas_lgbm_train["Sensibilidade"],
    "Especificidade": metricas_lgbm_train["Especificidade"],
    "VPP": metricas_lgbm_train["VPP"],
    "VPN": metricas_lgbm_train["VPN"],
    "Acuracia": metricas_lgbm_train["Acuracia"],
    "F1_Score": metricas_lgbm_train["F1_Score"],
    "TN": metricas_lgbm_train["TN"],
    "FP": metricas_lgbm_train["FP"],
    "FN": metricas_lgbm_train["FN"],
    "TP": metricas_lgbm_train["TP"]
}])

print("Avaliação do LightGBM (Base de Treino)")
print(metricas_lgbm_train_df)

print(f"\nAUC: {metricas_lgbm_train['AUC']:.3f}")
print(f"Ponto de corte (Youden): {metricas_lgbm_train['Ponto_Youden']:.3f}")
print(f"Sensibilidade: {metricas_lgbm_train['Sensibilidade']:.1%}")
print(f"Especificidade: {metricas_lgbm_train['Especificidade']:.1%}")
print(f"VPP: {metricas_lgbm_train['VPP']:.1%}")
print(f"VPN: {metricas_lgbm_train['VPN']:.1%}")
print(f"Acurácia: {metricas_lgbm_train['Acuracia']:.1%}")
print(f"F1 Score: {metricas_lgbm_train['F1_Score']:.3f}")
print(f"TN: {metricas_lgbm_train['TN']}")
print(f"FP: {metricas_lgbm_train['FP']}")
print(f"FN: {metricas_lgbm_train['FN']}")
print(f"TP: {metricas_lgbm_train['TP']}")

#%% Matriz de confusão e métricas (base de teste)

# Probabilidades e classes preditas com ponto de corte de Youden
metricas_lgbm_test, lgbm_pred_test_class = calcular_metricas_binarias(y_test, lgbm_pred_test_prob[:, 1])

# Matriz de confusão
lgbm_cm_test = confusion_matrix(y_test, lgbm_pred_test_class)
cm_lgbm_test = ConfusionMatrixDisplay(
    confusion_matrix=lgbm_cm_test,
    display_labels=['Sem Progressão', 'Com Progressão']
)

plt.figure(figsize=(8, 6), dpi=600)
cm_lgbm_test.plot(colorbar=False, cmap='Purples', values_format='d')
plt.title('LightGBM: Teste')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.tight_layout()
plt.show()

# Tabela de métricas
metricas_lgbm_test_df = pd.DataFrame([{
    "Base": "Teste",
    "AUC": metricas_lgbm_test["AUC"],
    "Ponto_Youden": metricas_lgbm_test["Ponto_Youden"],
    "Sensibilidade": metricas_lgbm_test["Sensibilidade"],
    "Especificidade": metricas_lgbm_test["Especificidade"],
    "VPP": metricas_lgbm_test["VPP"],
    "VPN": metricas_lgbm_test["VPN"],
    "Acuracia": metricas_lgbm_test["Acuracia"],
    "F1_Score": metricas_lgbm_test["F1_Score"],
    "TN": metricas_lgbm_test["TN"],
    "FP": metricas_lgbm_test["FP"],
    "FN": metricas_lgbm_test["FN"],
    "TP": metricas_lgbm_test["TP"]
}])

print("Avaliação do LightGBM (Base de Teste)")
print(metricas_lgbm_test_df)

print(f"\nAUC: {metricas_lgbm_test['AUC']:.3f}")
print(f"Ponto de corte (Youden): {metricas_lgbm_test['Ponto_Youden']:.3f}")
print(f"Sensibilidade: {metricas_lgbm_test['Sensibilidade']:.1%}")
print(f"Especificidade: {metricas_lgbm_test['Especificidade']:.1%}")
print(f"VPP: {metricas_lgbm_test['VPP']:.1%}")
print(f"VPN: {metricas_lgbm_test['VPN']:.1%}")
print(f"Acurácia: {metricas_lgbm_test['Acuracia']:.1%}")
print(f"F1 Score: {metricas_lgbm_test['F1_Score']:.3f}")
print(f"TN: {metricas_lgbm_test['TN']}")
print(f"FP: {metricas_lgbm_test['FP']}")
print(f"FN: {metricas_lgbm_test['FN']}")
print(f"TP: {metricas_lgbm_test['TP']}")

#%% Importância das variáveis preditoras

lgbm_features = pd.DataFrame({
    'features': [rotulos_legiveis_modelos[c] for c in X_tree.columns.tolist()],
    'importance': lgbm_best.feature_importances_
}).sort_values(by='importance', ascending=False)

print(lgbm_features)

# --------------------------------------------------
# Gráfico de importância das variáveis
# --------------------------------------------------
top_n = 10
lgbm_features_plot = lgbm_features.head(top_n)

plt.figure(figsize=(10, 6), dpi=600)
plt.barh(
    lgbm_features_plot["features"][::-1],
    lgbm_features_plot["importance"][::-1],
    color='mediumpurple'
)
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.title("Importância das Variáveis - LightGBM")
plt.tight_layout()
plt.show()

#%% V) XGBoost - Estimando um XGBoost

xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    colsample_bytree=0.8,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric='logloss',
    random_state=100
)
xgb_clf.fit(X_train, y_train)

#%% Grid Search para XGBoost

param_grid_xgb = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 4, 5, 6, None],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

xgb_grid = XGBClassifier(
    eval_metric='logloss',
    random_state=100
)

xgb_grid_model = GridSearchCV(
    estimator=xgb_grid,
    param_grid=param_grid_xgb,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

xgb_grid_model.fit(X_train, y_train)

print("Melhores parâmetros do XGBoost:")
print(xgb_grid_model.best_params_)

print("\nMelhor AUC média na validação cruzada:")
print(xgb_grid_model.best_score_)

xgb_best = xgb_grid_model.best_estimator_

#%% Obtendo os valores preditos pelo XGBoost

xgb_pred_train_class = xgb_best.predict(X_train)
xgb_pred_train_prob = xgb_best.predict_proba(X_train)

xgb_pred_test_class = xgb_best.predict(X_test)
xgb_pred_test_prob = xgb_best.predict_proba(X_test)

#%% Matriz de confusão e métricas (base de treino)

# Probabilidades e classes preditas com ponto de corte de Youden
metricas_xgb_train, xgb_pred_train_class = calcular_metricas_binarias(y_train, xgb_pred_train_prob[:, 1])

# Matriz de confusão
xgb_cm_train = confusion_matrix(y_train, xgb_pred_train_class)
cm_xgb_train = ConfusionMatrixDisplay(
    confusion_matrix=xgb_cm_train,
    display_labels=['Sem Progressão', 'Com Progressão']
)

plt.figure(figsize=(8, 6), dpi=600)
cm_xgb_train.plot(colorbar=False, cmap='summer', values_format='d')
plt.title('XGBoost: Treino')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.tight_layout()
plt.show()

# Tabela de métricas
metricas_xgb_train_df = pd.DataFrame([{
    "Base": "Treino",
    "AUC": metricas_xgb_train["AUC"],
    "Ponto_Youden": metricas_xgb_train["Ponto_Youden"],
    "Sensibilidade": metricas_xgb_train["Sensibilidade"],
    "Especificidade": metricas_xgb_train["Especificidade"],
    "VPP": metricas_xgb_train["VPP"],
    "VPN": metricas_xgb_train["VPN"],
    "Acuracia": metricas_xgb_train["Acuracia"],
    "F1_Score": metricas_xgb_train["F1_Score"],
    "TN": metricas_xgb_train["TN"],
    "FP": metricas_xgb_train["FP"],
    "FN": metricas_xgb_train["FN"],
    "TP": metricas_xgb_train["TP"]
}])

print("Avaliação do XGBoost (Base de Treino)")
print(metricas_xgb_train_df)

print(f"\nAUC: {metricas_xgb_train['AUC']:.3f}")
print(f"Ponto de corte (Youden): {metricas_xgb_train['Ponto_Youden']:.3f}")
print(f"Sensibilidade: {metricas_xgb_train['Sensibilidade']:.1%}")
print(f"Especificidade: {metricas_xgb_train['Especificidade']:.1%}")
print(f"VPP: {metricas_xgb_train['VPP']:.1%}")
print(f"VPN: {metricas_xgb_train['VPN']:.1%}")
print(f"Acurácia: {metricas_xgb_train['Acuracia']:.1%}")
print(f"F1 Score: {metricas_xgb_train['F1_Score']:.3f}")
print(f"TN: {metricas_xgb_train['TN']}")
print(f"FP: {metricas_xgb_train['FP']}")
print(f"FN: {metricas_xgb_train['FN']}")
print(f"TP: {metricas_xgb_train['TP']}")

#%% Matriz de confusão e métricas (base de teste)

# Probabilidades e classes preditas com ponto de corte de Youden
metricas_xgb_test, xgb_pred_test_class = calcular_metricas_binarias(y_test, xgb_pred_test_prob[:, 1])

# Matriz de confusão
xgb_cm_test = confusion_matrix(y_test, xgb_pred_test_class)
cm_xgb_test = ConfusionMatrixDisplay(
    confusion_matrix=xgb_cm_test,
    display_labels=['Sem Progressão', 'Com Progressão']
)

plt.figure(figsize=(8, 6), dpi=600)
cm_xgb_test.plot(colorbar=False, cmap='summer', values_format='d')
plt.title('XGBoost: Teste')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.tight_layout()
plt.show()

# Tabela de métricas
metricas_xgb_test_df = pd.DataFrame([{
    "Base": "Teste",
    "AUC": metricas_xgb_test["AUC"],
    "Ponto_Youden": metricas_xgb_test["Ponto_Youden"],
    "Sensibilidade": metricas_xgb_test["Sensibilidade"],
    "Especificidade": metricas_xgb_test["Especificidade"],
    "VPP": metricas_xgb_test["VPP"],
    "VPN": metricas_xgb_test["VPN"],
    "Acuracia": metricas_xgb_test["Acuracia"],
    "F1_Score": metricas_xgb_test["F1_Score"],
    "TN": metricas_xgb_test["TN"],
    "FP": metricas_xgb_test["FP"],
    "FN": metricas_xgb_test["FN"],
    "TP": metricas_xgb_test["TP"]
}])

print("Avaliação do XGBoost (Base de Teste)")
print(metricas_xgb_test_df)

print(f"\nAUC: {metricas_xgb_test['AUC']:.3f}")
print(f"Ponto de corte (Youden): {metricas_xgb_test['Ponto_Youden']:.3f}")
print(f"Sensibilidade: {metricas_xgb_test['Sensibilidade']:.1%}")
print(f"Especificidade: {metricas_xgb_test['Especificidade']:.1%}")
print(f"VPP: {metricas_xgb_test['VPP']:.1%}")
print(f"VPN: {metricas_xgb_test['VPN']:.1%}")
print(f"Acurácia: {metricas_xgb_test['Acuracia']:.1%}")
print(f"F1 Score: {metricas_xgb_test['F1_Score']:.3f}")
print(f"TN: {metricas_xgb_test['TN']}")
print(f"FP: {metricas_xgb_test['FP']}")
print(f"FN: {metricas_xgb_test['FN']}")
print(f"TP: {metricas_xgb_test['TP']}")

#%% Importância das variáveis preditoras

xgb_features = pd.DataFrame({
    'feature_original': X_tree.columns.tolist(),
    'feature_label': [rotulos_legiveis_modelos[c] for c in X_tree.columns.tolist()],
    'importance': xgb_best.feature_importances_
}).sort_values(by='importance', ascending=False)

print(xgb_features)

# --------------------------------------------------
# Gráfico de importância das variáveis
# --------------------------------------------------
top_n = 10
xgb_features_plot = xgb_features.head(top_n)

plt.figure(figsize=(10, 6), dpi=600)
plt.barh(
    xgb_features_plot["feature_label"][::-1],
    xgb_features_plot["importance"][::-1],
    color='forestgreen'
)
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.title("Importância das Variáveis - XGBoost")
plt.tight_layout()
plt.show()

#%% Figura com a importância das variáveis de cada modelo

def coluna_plot_features(df):
    if "feature_label" in df.columns:
        return "feature_label"
    elif "features" in df.columns:
        return "features"
    elif "feature_original" in df.columns:
        return "feature_original"
    else:
        raise KeyError("Nenhuma coluna de nomes de variáveis encontrada no dataframe.")

top_n = 10
fig, axes = plt.subplots(2, 2, figsize=(22, 16), dpi=600)
axes = axes.flatten()

tree_features_plot = tree_features.head(top_n)
rf_features_plot = rf_features.head(top_n)
lgbm_features_plot = lgbm_features.head(top_n)
xgb_features_plot = xgb_features.head(top_n)

col_tree = coluna_plot_features(tree_features_plot)
col_rf = coluna_plot_features(rf_features_plot)
col_lgbm = coluna_plot_features(lgbm_features_plot)
col_xgb = coluna_plot_features(xgb_features_plot)

# Árvore
axes[0].barh(tree_features_plot[col_tree][::-1], tree_features_plot["importance"][::-1], color='steelblue')
axes[0].set_title('Importância das Variáveis - Árvore Inicial')
axes[0].set_xlabel('Importância')
axes[0].set_ylabel('Variáveis')

# Random Forest
axes[1].barh(rf_features_plot[col_rf][::-1], rf_features_plot["importance"][::-1], color='darkorange')
axes[1].set_title('Importância das Variáveis - Random Forest')
axes[1].set_xlabel('Importância')
axes[1].set_ylabel('Variáveis')

# LightGBM
axes[2].barh(lgbm_features_plot[col_lgbm][::-1], lgbm_features_plot["importance"][::-1], color='mediumpurple')
axes[2].set_title('Importância das Variáveis - LightGBM')
axes[2].set_xlabel('Importância')
axes[2].set_ylabel('Variáveis')

# XGBoost
axes[3].barh(xgb_features_plot[col_xgb][::-1], xgb_features_plot["importance"][::-1], color='forestgreen')
axes[3].set_title('Importância das Variáveis - XGBoost')
axes[3].set_xlabel('Importância')
axes[3].set_ylabel('Variáveis')

plt.tight_layout()
plt.show()

#%% Curvas ROC - árvore ajustada, Random Forest, LightGBM e XGBoost

# Árvore ajustada
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, tree_pred_test_prob[:, 1])
roc_auc_tree = auc(fpr_tree, tpr_tree)

# Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_pred_test_prob[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

# LightGBM
fpr_lgbm, tpr_lgbm, thresholds_lgbm = roc_curve(y_test, lgbm_pred_test_prob[:, 1])
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

# XGBoost
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, xgb_pred_test_prob[:, 1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(15, 10), dpi=600)
plt.plot(fpr_tree, tpr_tree, linewidth=3, label=f'Árvore Ajustada (AUC = {roc_auc_tree:.3f})')
plt.plot(fpr_rf, tpr_rf, linewidth=3, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot(fpr_lgbm, tpr_lgbm, linewidth=3, label=f'LightGBM (AUC = {roc_auc_lgbm:.3f})')
plt.plot(fpr_xgb, tpr_xgb, linewidth=3, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed')

plt.title('Curvas ROC dos Modelos', fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensibilidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(fontsize=13, loc='lower right')
plt.show()

#%% SHAP do XGBoost - variáveis importantes do modelo

# Top variáveis mais importantes do XGBoost
top_n_shap = 10
top_features_xgb = xgb_features.head(top_n_shap)['feature_original'].tolist()

X_train_xgb_top = X_train[top_features_xgb].copy()
X_test_xgb_top = X_test[top_features_xgb].copy()

# Reajuste de um XGBoost apenas com as variáveis mais importantes
xgb_shap = XGBClassifier(
    n_estimators=xgb_best.get_params()['n_estimators'],
    max_depth=xgb_best.get_params()['max_depth'],
    colsample_bytree=xgb_best.get_params()['colsample_bytree'],
    learning_rate=xgb_best.get_params()['learning_rate'],
    subsample=xgb_best.get_params().get('subsample', 1.0),
    eval_metric='logloss',
    random_state=100
)

xgb_shap.fit(X_train_xgb_top, y_train)

# SHAP values
explainer_xgb = shap.TreeExplainer(xgb_shap)
shap_values_xgb = explainer_xgb.shap_values(X_test_xgb_top)

print("Variáveis utilizadas no SHAP:")
print(top_features_xgb)

#%% SHAP summary plot (beeswarm)

X_test_xgb_top_plot = X_test_xgb_top.rename(columns=rotulos_legiveis_modelos)

plt.figure(figsize=(14, 8), dpi=600)
shap.summary_plot(
    shap_values_xgb,
    X_test_xgb_top,
    plot_type='dot',
    show=False
)
plt.title('SHAP Summary Plot - XGBoost')
plt.tight_layout()
plt.show()

#%% SHAP bar plot

plt.figure(figsize=(14, 8), dpi=600)
shap.summary_plot(
    shap_values_xgb,
    X_test_xgb_top,
    plot_type='bar',
    show=False
)
plt.title('SHAP Feature Importance - XGBoost')
plt.tight_layout()
plt.show()

#%% VI) Rede Neural - Preparação dos dados

# Se X_tree ainda não existir, recria:
X_tree = pd.get_dummies(X, drop_first=True, dtype=int)

# Garantir y binário inteiro
y_nn = pd.to_numeric(y, errors="coerce").astype(int)

# Holdout treino/teste
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_tree,
    y_nn,
    test_size=0.40,
    random_state=100,
    stratify=y_nn
)

# Padronização
scaler_nn = StandardScaler()
X_train_nn_sc = scaler_nn.fit_transform(X_train_nn)
X_test_nn_sc = scaler_nn.transform(X_test_nn)

print("Dimensão treino:", X_train_nn_sc.shape)
print("Dimensão teste:", X_test_nn_sc.shape)

#%% GridSearch para Rede Neural com SciKeras

def criar_modelo_nn(n_hidden_1=16, n_hidden_2=8, dropout=0.0, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(n_hidden_1, activation="relu", input_shape=(X_train_nn_sc.shape[1],)))
    
    if dropout > 0:
        model.add(Dropout(dropout))
    
    if n_hidden_2 > 0:
        model.add(Dense(n_hidden_2, activation="relu"))
        if dropout > 0:
            model.add(Dropout(dropout))
    
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["AUC"]
    )
    return model

nn_clf = KerasClassifier(
    model=criar_modelo_nn,
    verbose=0
)

param_grid_nn = {
    "model__n_hidden_1": [8, 16, 32],
    "model__n_hidden_2": [0, 8, 16],
    "model__dropout": [0.0, 0.2],
    "model__learning_rate": [0.001, 0.0005],
    "batch_size": [16, 32],
    "epochs": [50, 100]
}

grid_nn = GridSearchCV(
    estimator=nn_clf,
    param_grid=param_grid_nn,
    scoring="roc_auc",
    cv=5,
    n_jobs=1,
    verbose=2
)

grid_nn.fit(X_train_nn_sc, y_train_nn)

print("Melhores parâmetros da rede neural:")
print(grid_nn.best_params_)

print("\nMelhor AUC média na validação cruzada:")
print(grid_nn.best_score_)

# Melhor modelo
nn_best = grid_nn.best_estimator_

# Probabilidades
nn_best_prob_train = nn_best.predict_proba(X_train_nn_sc)[:, 1]
nn_best_prob_test = nn_best.predict_proba(X_test_nn_sc)[:, 1]

metricas_nn_grid_train, pred_nn_grid_train = calcular_metricas_binarias(y_train_nn, nn_best_prob_train)
metricas_nn_grid_test, pred_nn_grid_test = calcular_metricas_binarias(y_test_nn, nn_best_prob_test)

print("\nMétricas da Rede Neural - treino")
print(metricas_nn_grid_train)

print("\nMétricas da Rede Neural - teste")
print(metricas_nn_grid_test)

#%% Construção e treino da rede neural

# -------------------------------
# Checagem de consistência
# -------------------------------
print("X_train_nn_sc:", X_train_nn_sc.shape)
print("X_test_nn_sc :", X_test_nn_sc.shape)
print("y_train_nn   :", np.asarray(y_train_nn).shape)
print("y_test_nn    :", np.asarray(y_test_nn).shape)

if X_train_nn_sc.shape[0] != len(y_train_nn):
    raise ValueError(
        f"Inconsistência no treino: X_train_nn_sc tem {X_train_nn_sc.shape[0]} linhas "
        f"e y_train_nn tem {len(y_train_nn)} observações."
    )

if X_test_nn_sc.shape[0] != len(y_test_nn):
    raise ValueError(
        f"Inconsistência no teste: X_test_nn_sc tem {X_test_nn_sc.shape[0]} linhas "
        f"e y_test_nn tem {len(y_test_nn)} observações."
    )

model_nn = Sequential([
    Dense(8, activation='relu', input_shape=(X_train_nn_sc.shape[1],)),
    Dense(1, activation='sigmoid')
])

model_nn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_nn = model_nn.fit(
    X_train_nn_sc,
    np.asarray(y_train_nn),
    epochs=50,
    batch_size=16,
    validation_data=(X_test_nn_sc, np.asarray(y_test_nn)),
    verbose=1
)

model_nn.summary()

#%% Avaliação da Rede Neural

# Probabilidades previstas
nn_pred_train_prob = model_nn.predict(X_train_nn_sc, verbose=0).ravel()
nn_pred_test_prob  = model_nn.predict(X_test_nn_sc, verbose=0).ravel()

# Métricas pelo ponto de Youden
metricas_nn_train, nn_pred_train_class = calcular_metricas_binarias(
    np.asarray(y_train_nn),
    nn_pred_train_prob
)

metricas_nn_test, nn_pred_test_class = calcular_metricas_binarias(
    np.asarray(y_test_nn),
    nn_pred_test_prob
)

print("=" * 80)
print("REDE NEURAL - BASE DE TREINO")
print("=" * 80)
for k, v in metricas_nn_train.items():
    print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")

print("\n" + "=" * 80)
print("REDE NEURAL - BASE DE TESTE")
print("=" * 80)
for k, v in metricas_nn_test.items():
    print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")

auc_train_nn = roc_auc_score(np.asarray(y_train_nn), nn_pred_train_prob)
auc_test_nn  = roc_auc_score(np.asarray(y_test_nn), nn_pred_test_prob)
auc_test_inv = roc_auc_score(np.asarray(y_test_nn), 1 - nn_pred_test_prob)

print("\nAUC treino:", round(auc_train_nn, 4))
print("AUC teste:", round(auc_test_nn, 4))
print("AUC teste invertida:", round(auc_test_inv, 4))

#%% Matrizes de Confusão da Rede Neural

# Treino
nn_cm_train = confusion_matrix(np.asarray(y_train_nn), nn_pred_train_class)
disp_nn_train = ConfusionMatrixDisplay(confusion_matrix=nn_cm_train)

plt.rcParams['figure.dpi'] = 600
disp_nn_train.plot(colorbar=False, cmap='Purples')
plt.title('Rede Neural: Treino')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.show()

# Teste
nn_cm_test = confusion_matrix(np.asarray(y_test_nn), nn_pred_test_class)
disp_nn_test = ConfusionMatrixDisplay(confusion_matrix=nn_cm_test)

plt.rcParams['figure.dpi'] = 600
disp_nn_test.plot(colorbar=False, cmap='Purples')
plt.title('Rede Neural: Teste')
plt.xlabel('Classificado (Modelo)')
plt.ylabel('Observado (Real)')
plt.show()

#%% Curva ROC da Rede Neural

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr_nn, tpr_nn, thresholds_nn = roc_curve(np.asarray(y_test_nn), nn_pred_test_prob)
roc_auc_nn = auc(fpr_nn, tpr_nn)

plt.figure(figsize=(15, 10), dpi=600)
plt.plot(fpr_nn, tpr_nn, color='purple', linewidth=4)
plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed')
plt.title(f'AUC-ROC Rede Neural: {roc_auc_nn:.3f}', fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensibilidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

print(f"AUC da Rede Neural (teste): {roc_auc_nn:.4f}")

#%% Probabilidade da Rede Neural para toda a base

if 'X_nn' not in globals():
    raise NameError(
        "A variável 'X_nn' não foi encontrada. "
        "Crie X_nn com as mesmas colunas usadas no treino da rede neural."
    )

X_nn_all = scaler_nn.transform(X_nn)
dados["proba_rede_neural"] = model_nn.predict(X_nn_all, verbose=0).ravel()

# Classes pelo ponto de Youden obtido na base de teste
corte_nn = metricas_nn_test["Ponto_Youden"]
dados["pred_rede_neural"] = (dados["proba_rede_neural"] >= corte_nn).astype(int)

print(dados[["proba_rede_neural", "pred_rede_neural"]].head())

#%% Histórico de Treino da Rede Neural

historico_nn = pd.DataFrame(history_nn.history)
historico_nn["epoch"] = historico_nn.index + 1

display(historico_nn.head())

#%% Métricas consolidadas da Rede Neural

metricas_rede_neural = pd.DataFrame([
    {
        "Base": "Treino",
        **metricas_nn_train
    },
    {
        "Base": "Teste",
        **metricas_nn_test
    }
])

print(metricas_rede_neural)

#%% Salvar resultados completos em Excel e figuras

nome_arquivo_saida = arquivo_excel_saida

# ------------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------------
def salvar_figura_png(fig, nome_arquivo):
    caminho = PASTA_IMAGENS / nome_arquivo
    fig.savefig(caminho, dpi=600, bbox_inches="tight")
    return str(caminho)

def sheet_name_seguro(nome):
    nome = str(nome).replace("/", "_").replace("\\", "_").replace(":", "_")
    return nome[:31]

def padronizar_df_importancia(df):
    """
    Padroniza DataFrames de importância para colunas:
    - features
    - importance
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df2 = df.copy()

    mapa_rename = {}
    for col in df2.columns:
        col_lower = str(col).strip().lower()
        if col_lower in ["feature", "features", "variavel", "variáveis", "variavelis", "variaveis"]:
            mapa_rename[col] = "features"
        elif col_lower in ["importance", "importancia", "importância"]:
            mapa_rename[col] = "importance"

    df2 = df2.rename(columns=mapa_rename)

    if "features" not in df2.columns and df2.shape[1] >= 1:
        df2 = df2.rename(columns={df2.columns[0]: "features"})

    if "importance" not in df2.columns and df2.shape[1] >= 2:
        df2 = df2.rename(columns={df2.columns[1]: "importance"})

    if not {"features", "importance"}.issubset(df2.columns):
        return None

    df2 = df2[["features", "importance"]].copy()
    return df2

arquivos_figuras = []

# ------------------------------------------------------------------
# Base de resultados consolidada
# ------------------------------------------------------------------
dados_resultados = dados.copy()

for col in [
    "proba_arvore", "pred_arvore",
    "proba_rede_neural", "pred_rede_neural"
]:
    if col in dados.columns:
        dados_resultados[col] = dados[col]

if "proba_logit_multi" in locals():
    try:
        dados_resultados["proba_logit_multi"] = np.asarray(proba_logit_multi)
    except Exception:
        pass

if "pred_logit_multi" in locals():
    try:
        dados_resultados["pred_logit_multi"] = np.asarray(pred_logit_multi)
    except Exception:
        pass

# ------------------------------------------------------------------
# Métricas consolidadas dos modelos
# ------------------------------------------------------------------
metricas_modelos = []

if "metricas_logit_multi" in locals() and isinstance(metricas_logit_multi, dict):
    metricas_modelos.append({
        "Modelo": "Regressão Logística Multivariada",
        "Base": "Completa",
        **metricas_logit_multi
    })

if "metricas_tree_grid_train" in locals() and isinstance(metricas_tree_grid_train, dict):
    metricas_modelos.append({
        "Modelo": "Árvore de Decisão",
        "Base": "Treino",
        **metricas_tree_grid_train
    })

if "metricas_tree_grid_test" in locals() and isinstance(metricas_tree_grid_test, dict):
    metricas_modelos.append({
        "Modelo": "Árvore de Decisão",
        "Base": "Teste",
        **metricas_tree_grid_test
    })

if "metricas_rf_train" in locals() and isinstance(metricas_rf_train, dict):
    metricas_modelos.append({
        "Modelo": "Random Forest",
        "Base": "Treino",
        **metricas_rf_train
    })

if "metricas_rf_test" in locals() and isinstance(metricas_rf_test, dict):
    metricas_modelos.append({
        "Modelo": "Random Forest",
        "Base": "Teste",
        **metricas_rf_test
    })

if "metricas_lgbm_train" in locals() and isinstance(metricas_lgbm_train, dict):
    metricas_modelos.append({
        "Modelo": "LightGBM",
        "Base": "Treino",
        **metricas_lgbm_train
    })

if "metricas_lgbm_test" in locals() and isinstance(metricas_lgbm_test, dict):
    metricas_modelos.append({
        "Modelo": "LightGBM",
        "Base": "Teste",
        **metricas_lgbm_test
    })

if "metricas_xgb_train" in locals() and isinstance(metricas_xgb_train, dict):
    metricas_modelos.append({
        "Modelo": "XGBoost",
        "Base": "Treino",
        **metricas_xgb_train
    })

if "metricas_xgb_test" in locals() and isinstance(metricas_xgb_test, dict):
    metricas_modelos.append({
        "Modelo": "XGBoost",
        "Base": "Teste",
        **metricas_xgb_test
    })

if "metricas_rede_neural" in locals() and isinstance(metricas_rede_neural, pd.DataFrame) and not metricas_rede_neural.empty:
    for _, row in metricas_rede_neural.iterrows():
        registro = row.to_dict()
        registro["Modelo"] = "Rede Neural"
        metricas_modelos.append(registro)

metricas_modelos = pd.DataFrame(metricas_modelos)

# ------------------------------------------------------------------
# Padronização dos DataFrames de importância
# ------------------------------------------------------------------
tree_features_exp = padronizar_df_importancia(tree_features) if "tree_features" in locals() else None
rf_features_exp = padronizar_df_importancia(rf_features) if "rf_features" in locals() else None
lgbm_features_exp = padronizar_df_importancia(lgbm_features) if "lgbm_features" in locals() else None
xgb_features_exp = padronizar_df_importancia(xgb_features) if "xgb_features" in locals() else None

# ------------------------------------------------------------------
# Salvar figuras principais em PNG
# ------------------------------------------------------------------

# 1) Árvore inicial
if all(v in locals() for v in ["tree_clf", "X_tree"]):
    fig = plt.figure(figsize=(16, 8), dpi=600)
    plot_tree(
        tree_clf,
        feature_names=X_tree.columns.tolist(),
        class_names=['Sem Progressão', 'Com Progressão'],
        proportion=False,
        filled=True,
        node_ids=True,
        rounded=True,
        fontsize=16,
        impurity=True,
        label='all'
    )
    plt.tight_layout(pad=1.5)
    caminho = salvar_figura_png(fig, "arvore_inicial.png")
    arquivos_figuras.append({"Figura": "Árvore inicial", "Arquivo": caminho})
    plt.close(fig)

# 2) Importância individual de cada modelo
if tree_features_exp is not None:
    fig = plt.figure(figsize=(10, 6), dpi=600)
    df_plot = tree_features_exp.head(10)
    plt.barh(df_plot["features"][::-1], df_plot["importance"][::-1])
    plt.xlabel("Importância")
    plt.ylabel("Variáveis")
    plt.title("Importância das Variáveis - Árvore de Decisão")
    plt.tight_layout()
    caminho = salvar_figura_png(fig, "importancia_variaveis_arvore.png")
    arquivos_figuras.append({"Figura": "Importância das variáveis - Árvore", "Arquivo": caminho})
    plt.close(fig)

if rf_features_exp is not None:
    fig = plt.figure(figsize=(10, 6), dpi=600)
    df_plot = rf_features_exp.head(10)
    plt.barh(df_plot["features"][::-1], df_plot["importance"][::-1], color="darkorange")
    plt.xlabel("Importância")
    plt.ylabel("Variáveis")
    plt.title("Importância das Variáveis - Random Forest")
    plt.tight_layout()
    caminho = salvar_figura_png(fig, "importancia_variaveis_rf.png")
    arquivos_figuras.append({"Figura": "Importância das variáveis - Random Forest", "Arquivo": caminho})
    plt.close(fig)

if lgbm_features_exp is not None:
    fig = plt.figure(figsize=(10, 6), dpi=600)
    df_plot = lgbm_features_exp.head(10)
    plt.barh(df_plot["features"][::-1], df_plot["importance"][::-1], color="mediumpurple")
    plt.xlabel("Importância")
    plt.ylabel("Variáveis")
    plt.title("Importância das Variáveis - LightGBM")
    plt.tight_layout()
    caminho = salvar_figura_png(fig, "importancia_variaveis_lgbm.png")
    arquivos_figuras.append({"Figura": "Importância das variáveis - LightGBM", "Arquivo": caminho})
    plt.close(fig)

if xgb_features_exp is not None:
    fig = plt.figure(figsize=(10, 6), dpi=600)
    df_plot = xgb_features_exp.head(10)
    plt.barh(df_plot["features"][::-1], df_plot["importance"][::-1], color="forestgreen")
    plt.xlabel("Importância")
    plt.ylabel("Variáveis")
    plt.title("Importância das Variáveis - XGBoost")
    plt.tight_layout()
    caminho = salvar_figura_png(fig, "importancia_variaveis_xgb.png")
    arquivos_figuras.append({"Figura": "Importância das variáveis - XGBoost", "Arquivo": caminho})
    plt.close(fig)

# 3) Figura combinada das importâncias
if any(df is not None for df in [tree_features_exp, rf_features_exp, lgbm_features_exp, xgb_features_exp]):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16), dpi=600)
    axes = axes.flatten()

    lista_modelos = [
        ("Árvore de Decisão", tree_features_exp, None),
        ("Random Forest", rf_features_exp, "darkorange"),
        ("LightGBM", lgbm_features_exp, "mediumpurple"),
        ("XGBoost", xgb_features_exp, "forestgreen"),
    ]

    for ax, (titulo, df_imp, cor) in zip(axes, lista_modelos):
        if df_imp is not None and not df_imp.empty:
            df_plot = df_imp.head(10)
            ax.barh(df_plot["features"][::-1], df_plot["importance"][::-1], color=cor)
            ax.set_title(f"Importância das Variáveis - {titulo}")
            ax.set_xlabel("Importância")
            ax.set_ylabel("Variáveis")
        else:
            ax.axis("off")

    plt.tight_layout()
    caminho = salvar_figura_png(fig, "importancia_variaveis_modelos.png")
    arquivos_figuras.append({"Figura": "Importância das variáveis dos modelos", "Arquivo": caminho})
    plt.close(fig)

# 4) Curvas ROC dos modelos
if all(v in locals() for v in ["fpr_tree", "tpr_tree", "roc_auc_tree",
                               "fpr_rf", "tpr_rf", "roc_auc_rf",
                               "fpr_lgbm", "tpr_lgbm", "roc_auc_lgbm",
                               "fpr_xgb", "tpr_xgb", "roc_auc_xgb"]):
    fig = plt.figure(figsize=(15, 10), dpi=600)
    plt.plot(fpr_tree, tpr_tree, linewidth=3, label=f'Árvore de Decisão (AUC = {roc_auc_tree:.3f})')
    plt.plot(fpr_rf, tpr_rf, linewidth=3, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
    plt.plot(fpr_lgbm, tpr_lgbm, linewidth=3, label=f'LightGBM (AUC = {roc_auc_lgbm:.3f})')
    plt.plot(fpr_xgb, tpr_xgb, linewidth=3, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed')
    plt.title('Curvas ROC dos Modelos', fontsize=22)
    plt.xlabel('1 - Especificidade', fontsize=20)
    plt.ylabel('Sensibilidade', fontsize=20)
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.legend(fontsize=13, loc='lower right')
    plt.tight_layout()
    caminho = salvar_figura_png(fig, "roc_modelos.png")
    arquivos_figuras.append({"Figura": "Curvas ROC dos modelos", "Arquivo": caminho})
    plt.close(fig)

# 5) SHAP XGBoost
if all(v in locals() for v in ["shap_values_xgb", "X_test_xgb_top"]):
    shap.summary_plot(
        shap_values_xgb,
        X_test_xgb_top,
        plot_type='dot',
        show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    caminho = salvar_figura_png(fig, "shap_summary_xgboost.png")
    arquivos_figuras.append({"Figura": "SHAP summary plot - XGBoost", "Arquivo": caminho})
    plt.close(fig)

    shap.summary_plot(
        shap_values_xgb,
        X_test_xgb_top,
        plot_type='bar',
        show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    caminho = salvar_figura_png(fig, "shap_bar_xgboost.png")
    arquivos_figuras.append({"Figura": "SHAP bar plot - XGBoost", "Arquivo": caminho})
    plt.close(fig)

# 6) Curva ROC da rede neural
if all(v in locals() for v in ["fpr_nn", "tpr_nn", "roc_auc_nn"]):
    fig = plt.figure(figsize=(15, 10), dpi=600)
    plt.plot(fpr_nn, tpr_nn, color='purple', linewidth=4)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed')
    plt.title(f'AUC-ROC Rede Neural: {roc_auc_nn:.3f}', fontsize=22)
    plt.xlabel('1 - Especificidade', fontsize=20)
    plt.ylabel('Sensibilidade', fontsize=20)
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
    plt.tight_layout()
    caminho = salvar_figura_png(fig, "roc_rede_neural.png")
    arquivos_figuras.append({"Figura": "Curva ROC da rede neural", "Arquivo": caminho})
    plt.close(fig)

# 7) Histórico da rede neural
if "historico_nn" in locals() and isinstance(historico_nn, pd.DataFrame) and not historico_nn.empty:
    colunas_plot = [c for c in ["loss", "val_loss", "accuracy", "val_accuracy"] if c in historico_nn.columns]
    if len(colunas_plot) > 0:
        fig = plt.figure(figsize=(14, 8), dpi=600)
        for col in colunas_plot:
            plt.plot(historico_nn["epoch"], historico_nn[col], linewidth=2, label=col)
        plt.title("Histórico de Treino da Rede Neural")
        plt.xlabel("Época")
        plt.ylabel("Valor")
        plt.legend()
        plt.tight_layout()
        caminho = salvar_figura_png(fig, "historico_rede_neural.png")
        arquivos_figuras.append({"Figura": "Histórico da rede neural", "Arquivo": caminho})
        plt.close(fig)

# ------------------------------------------------------------------
# Exportação completa para Excel
# ------------------------------------------------------------------
with pd.ExcelWriter(nome_arquivo_saida, engine="openpyxl") as writer:

    # Base principal
    dados_resultados.to_excel(writer, sheet_name=sheet_name_seguro("Dados_resultados"), index=False)

    # Descritivos e testes
    if "resultados_testes" in locals() and isinstance(resultados_testes, pd.DataFrame):
        resultados_testes.to_excel(writer, sheet_name=sheet_name_seguro("Testes_normalidade"), index=False)

    if "resultados_numericos" in locals() and isinstance(resultados_numericos, pd.DataFrame):
        resultados_numericos.to_excel(writer, sheet_name=sheet_name_seguro("Tabela1_numericas"), index=False)

    if "resultados_categoricos" in locals() and isinstance(resultados_categoricos, pd.DataFrame):
        resultados_categoricos.to_excel(writer, sheet_name=sheet_name_seguro("Tabela1_categoricas"), index=False)

    if "resumo_numericas_por_teste" in locals() and isinstance(resumo_numericas_por_teste, pd.DataFrame):
        resumo_numericas_por_teste.to_excel(writer, sheet_name=sheet_name_seguro("Resumo_numericas"), index=False)

    if "resumo_diagnostico" in locals() and isinstance(resumo_diagnostico, pd.DataFrame):
        resumo_diagnostico.to_excel(writer, sheet_name=sheet_name_seguro("Diagnostico_categoricas"), index=False)

    # Regressão logística
    if "resultados_logit_uni" in locals() and isinstance(resultados_logit_uni, pd.DataFrame):
        resultados_logit_uni.to_excel(writer, sheet_name=sheet_name_seguro("Logistica_univariada"), index=False)

    if "probs_univariadas" in locals() and isinstance(probs_univariadas, pd.DataFrame):
        probs_univariadas.to_excel(writer, sheet_name=sheet_name_seguro("Prob_univariadas"), index=True)

    if "resultado_logit_multi" in locals() and isinstance(resultado_logit_multi, pd.DataFrame):
        resultado_logit_multi.to_excel(writer, sheet_name=sheet_name_seguro("Logistica_multivariada"), index=False)

    if "resultados_variaveis_multi" in locals() and isinstance(resultados_variaveis_multi, pd.DataFrame):
        resultados_variaveis_multi.to_excel(writer, sheet_name=sheet_name_seguro("Logit_variaveis_multi"), index=False)

    if "qualidade_logit_multi" in locals() and isinstance(qualidade_logit_multi, pd.DataFrame):
        qualidade_logit_multi.to_excel(writer, sheet_name=sheet_name_seguro("Qualidade_logit_multi"), index=False)

    if "tabela_hl_multi" in locals() and isinstance(tabela_hl_multi, pd.DataFrame):
        tabela_hl_multi.to_excel(writer, sheet_name=sheet_name_seguro("Hosmer_Lemeshow"), index=False)

    # Importâncias
    if tree_features_exp is not None:
        tree_features_exp.to_excel(writer, sheet_name=sheet_name_seguro("Imp_arvore"), index=False)

    if rf_features_exp is not None:
        rf_features_exp.to_excel(writer, sheet_name=sheet_name_seguro("Imp_random_forest"), index=False)

    if lgbm_features_exp is not None:
        lgbm_features_exp.to_excel(writer, sheet_name=sheet_name_seguro("Imp_lightgbm"), index=False)

    if xgb_features_exp is not None:
        xgb_features_exp.to_excel(writer, sheet_name=sheet_name_seguro("Imp_xgboost"), index=False)

    # Grid search
    if "grid_tree" in locals():
        pd.DataFrame(grid_tree.cv_results_).to_excel(writer, sheet_name=sheet_name_seguro("Grid_tree"), index=False)

    if "rf_grid_model" in locals():
        pd.DataFrame(rf_grid_model.cv_results_).to_excel(writer, sheet_name=sheet_name_seguro("Grid_random_forest"), index=False)

    if "lgbm_grid_model" in locals():
        pd.DataFrame(lgbm_grid_model.cv_results_).to_excel(writer, sheet_name=sheet_name_seguro("Grid_lightgbm"), index=False)

    if "xgb_grid_model" in locals():
        pd.DataFrame(xgb_grid_model.cv_results_).to_excel(writer, sheet_name=sheet_name_seguro("Grid_xgboost"), index=False)

    # Métricas por modelo
    if "metricas_rf_train_df" in locals():
        metricas_rf_train_df.to_excel(writer, sheet_name=sheet_name_seguro("Metricas_rf_treino"), index=False)

    if "metricas_rf_test_df" in locals():
        metricas_rf_test_df.to_excel(writer, sheet_name=sheet_name_seguro("Metricas_rf_teste"), index=False)

    if "metricas_lgbm_train_df" in locals():
        metricas_lgbm_train_df.to_excel(writer, sheet_name=sheet_name_seguro("Metricas_lgbm_treino"), index=False)

    if "metricas_lgbm_test_df" in locals():
        metricas_lgbm_test_df.to_excel(writer, sheet_name=sheet_name_seguro("Metricas_lgbm_teste"), index=False)

    if "metricas_xgb_train_df" in locals():
        metricas_xgb_train_df.to_excel(writer, sheet_name=sheet_name_seguro("Metricas_xgb_treino"), index=False)

    if "metricas_xgb_test_df" in locals():
        metricas_xgb_test_df.to_excel(writer, sheet_name=sheet_name_seguro("Metricas_xgb_teste"), index=False)

    if "metricas_rede_neural" in locals() and isinstance(metricas_rede_neural, pd.DataFrame):
        metricas_rede_neural.to_excel(writer, sheet_name=sheet_name_seguro("Metricas_rede_neural"), index=False)

    if "historico_nn" in locals() and isinstance(historico_nn, pd.DataFrame):
        historico_nn.to_excel(writer, sheet_name=sheet_name_seguro("Historico_rede_neural"), index=False)

    if isinstance(metricas_modelos, pd.DataFrame) and not metricas_modelos.empty:
        metricas_modelos.to_excel(writer, sheet_name=sheet_name_seguro("Metricas_modelos"), index=False)

    # Índice dos arquivos de figura
    if len(arquivos_figuras) > 0:
        pd.DataFrame(arquivos_figuras).to_excel(writer, sheet_name=sheet_name_seguro("Arquivos_figuras"), index=False)

print(f"Arquivo Excel salvo com sucesso: {nome_arquivo_saida}")
print(f"Figuras salvas em: {PASTA_IMAGENS}")

if len(arquivos_figuras) > 0:
    print("\nArquivos de figuras gerados:")
    for item in arquivos_figuras:
        print(f"- {item['Figura']}: {item['Arquivo']}")

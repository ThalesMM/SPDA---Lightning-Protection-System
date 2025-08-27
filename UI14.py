# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------------
INTERFACE GRÁFICA PARA GERENCIAMENTO DE RISCO DA ABNT NBR 5419-2
-------------------------------------------------------------------------------
Versão Final e Precisa v4.0 (Completa):
- Backend: Inclui o cálculo dos 4 tipos de risco (R1, R2, R3, R4).
- Lógica Condicional Robusta: A análise se adapta dinamicamente para incluir
  R2 (serviços públicos) e R3 (patrimônio cultural) com base nas seleções do usuário.
- Frontend: A tabela de resultados agora exibe dinamicamente as colunas de risco
  relevantes para a análise.
- Correção de Erros Críticos: Corrigido o mapeamento de inputs, o escopo de
  variáveis da UI e a lógica de cálculo/recomendação para garantir precisão e
  estabilidade.
-------------------------------------------------------------------------------
"""

import math
import numpy as np
import pandas as pd
import gradio as gr

# ==============================================================================
# SEÇÃO 1: TABELAS E FUNÇÕES DE CÁLCULO DA NORMA (MOTOR DE CÁLCULO)
# ==============================================================================

def A_D(L, W, H):
    return (L * W) + 2 * (3 * H) * (L + W) + np.pi * (3 * H)**2

def AD_(Hp):
    return np.pi * (3 * Hp)**2

def N_D(Ng, L, W, H, Cd):
    return Ng * A_D(L, W, H) * Cd * 1e-6

def N_M(Ng, D, L, W):
    return Ng * (2 * D * (L + W) + np.pi * D**2) * 1e-6

def N_L(Ng, Ct, Cf, Ce):
    return Ng * 1000 * Ct * Cf * Ce * 1e-6

def N_DJ(Ng, L_adj, W_adj, H_adj, Hmax, Cd_adj_valor, Ct):
    if H_adj<Hmax:
        A_dj = max(A_D(L_adj, W_adj, H_adj), AD_(Hmax))
    else:
        A_dj = A_D(L_adj, W_adj, H_adj)
    return Ng * A_dj * Cd_adj_valor * Ct * 1e-6

def N_I(Ng, Ci, Ce, Ct):
    return Ng * (1000 * Ci * Ce * Ct) * 1e-6

def Pa(Pta, Pb):
    return Pta * Pb

def Pc(Pcspd, Cld):
    return Pcspd * Cld

def Pu(Ptu, Peb, Pld, Cld):
    return Ptu * Peb * Pld * Cld

def Pw(Pspd, Pld, Cld):
    return Pspd * Pld * Cld

def Pv(Peb, Pld, Cld):
    return Peb * Pld * Cld

# --- Tabelas da Norma ---

service = ["-", "TV, linhas de sinais", "Gás, água, energia elétrica"]
TABLE_4_1_CD = {"Estrutura cercada por objetos mais altos": 0.25, "Estrutura cercada por objetos de mesma altura ou mais baixos": 0.5, "Estrutura isolada": 1, "Estrutura isolada em morro ou colina": 2}
TABLE_4_2_CT = {"Linha aérea de energia ou sinal": 1, "Linha aérea com transformador AT/BT na entrada": 0.2}
TABLE_4_3_CI = {"Aéreo": 1, "Enterrado": 0.5, "Enterrado em malha de aterramento": 0.01}
TABLE_4_4_CE = {"Rural": 1, "Suburbano": 0.5, "Urbano": 0.1, "Urbano com edifícios > 20m": 0.01}
TABLE_4_5_PTA = {"Sem medidas de proteção": 1, "Avisos de alerta": 0.1, "Isolação elétrica (3mm polietileno reticulado das partes expostas, e. g. condutores de descidas); ou equipotencialização efetiva do solo": 0.01, 'Restrições físicas ou estruturas do edifício utilizadas como subsistema de descida': 0}
TABLE_4_6_PB = {"-": 1, "I": 0.02, "II": 0.05, "III": 0.1, "IV": 0.2, 'I + estrutura metálica ou descida natural': 0.01, 'Cobertura metálica + estrutura metálica ou descida natural': 0.001}
TABLE_4_7_PSPD = {"-": 1, "I": 0.01, "II": 0.02, "III": 0.05, "IV": 0.05, 'DPS com características melhoradas': 0.005}
TABLE_4_8_CLI_CLD = {"Aérea/Enterrada sem blindagem": (1, 1), "De energia com neutro multi-aterrado": (0.2, 1), "Enterrada com blindagem enterrada não interligada": (0.3, 1), "Aérea com blindagem aérea não interligada": (0.1, 1), "Aérea/Enterrada com blindagem interligada": (0, 1), 'Nenhuma linha externa ou interfaces isolantes de acordo com a NBR 5419-4 (sem risco)': (0, 0)}
TABLE_4_9_K3 = {"Cabo com blindagem ou em duto metálico": 0.0001, "Cabo sem blindagem sem evitar grandes laços": 1, "Cabo sem blindagem evitando grandes laços": 0.2,"Cabo sem blindagem se preocupando com o roteamento evitando grandes laços": 0.01}
TABLE_4_10_PTU = {"-": 1, "Avisos visíveis de alerta": 0.1, "Isolação elétrica adequada": 0.001, 'Restrições físicas': 0}
TABLE_4_11_PEB = {"-": 1, "I": 0.01, "II": 0.02, "III": 0.05, "IV": 0.05, 'DPS com características melhoradas': 0.005}

def TABLE_4_12_PLD(UW, blindagem, RS):
    if blindagem == 'nenhum ou não interligada':
        return 1
    elif blindagem == 'blindado e interligada':
        if RS <= 1:
            if UW == 1: return 0.6
            elif UW == 1.5: return 0.4
            elif UW == 2.5: return 0.2
            elif UW == 4: return 0.04
            elif UW == 6: return 0.02
        elif RS <= 5:
            if UW == 1: return 0.9
            elif UW == 1.5: return 0.8
            elif UW == 2.5: return 0.6
            elif UW == 4: return 0.3
            elif UW == 6: return 0.1
        elif RS <= 20:
            if UW == 1: return 1
            elif UW == 1.5: return 1
            elif UW == 2.5: return 0.95
            elif UW == 4: return 0.9
            elif UW == 6: return 0.8
    return 1 # Default

def TABLE_4_13_PLI(UW,tipoDeLinha):
    if tipoDeLinha == "Energia":
        if UW == 1: return 1
        elif UW == 1.5: return 0.6
        elif UW == 2.5: return 0.3
        elif UW == 4: return 0.16
        elif UW == 6: return 0.1
    elif tipoDeLinha == "Sinal":
        if UW == 1: return 1
        elif UW == 1.5: return 0.5
        elif UW == 2.5: return 0.2
        elif UW == 4: return 0.08
        elif UW == 6: return 0.04
    return 1 # Default

def Ks4(Uw): return 1/Uw
def Pms(Ks3, Ks4): return pow(0.12 * 0.12 * Ks3 * Ks4, 2)

TABLE_5_2_L1_LF = {"Risco de explosão": 0.1, "Hospital, hotel, escola": 0.1, "Entretenimento, igreja, museu": 0.05, "Industrial, comercial": 0.02, "Outros": 0.01}
TABLE_5_2_L1_LO = {"Risco de explosão": 0.1, "Unidade de terapia intensiva": 0.01, "Outras partes de hospital": 0.001, "Aplicações gerais": 0.001}
TABLE_5_2_L1_LT = 0.01
TABLE_5_3_RT = {"Concreto/agricultura (resistividade <= 1 kΩ.m)": 1e-2, 'Mármore/ cerâmica (1 a 10 kΩ.m)': 1e-3, "Cascalho/carpete/tapete (10 a 100 kΩ.m)": 1e-4, "Asfalto/linóleo/piso de madeira (> 100 kΩ.m)": 1e-5}
TABLE_5_4_Rp = {"Sem medidas": 1, "Extintores, hidrantes, instalações operadas de forma manual, detecção de alarme manual/automática, rotas de escape, compartimento à prova de fogo": 0.2, "Supressão automática": 0.2}
TABLE_5_5_Rf = {"Explosívos sólidos ou Zonas 0, 20": 1, "Zonas 1, 21 ou alto risco": 0.1, "Risco ordinário": 0.01, "Zonas 2, 22 ou baixo risco": 0.001, "Nenhum risco de incêndio": 0}
TABLE_5_6_HZ = {"Nenhum": 1, "Baixo pânico (<100 pessoas)": 2, "Médio pânico/Dificuldade de evacuação (100-1000 pessoas)": 5, "Alto pânico (>1000 pessoas)": 10}
TABLE_5_12_LF_Lo_L4 = {"Agrícola (animais)": 0.01, "Perda de patrimônio cultural insubstituível": 1, "Hospital, industrial, museu": 0.5, "Hotel, escola, escritório, comercial": 0.2, "Outros": 0.1}
TABLE_5_12_Lo_L4_D3 = {"Hospital, industrial, escritório, hotel, comercial": 1, "Museu, agricultura, escola, igreja, entretenimento": 0.1, "Outros": 0.01}

def TABLE_5_1_L1(r_t, L_T, n_z, n_t, t_z, L_o, L_f, h_z, r_p, r_f):
    if n_t == 0 or t_z == 0: return {"D1": 0, "D2": 0, "D3": 0}
    f_o = (n_z / n_t) * (t_z / 8760)
    D1 = r_t * L_T * f_o
    D2 = r_p * r_f * h_z * L_f * f_o
    D3 = L_o * f_o
    return {"D1": D1, "D2": D2, "D3": D3}

def TABLE_5_7_L2(n_z, n_t, r_p, r_f, servico_fornecido):
    L_o, L_f = 0, 0
    if servico_fornecido == "TV, linhas de sinais": L_o, L_f = 0.001, 0.01
    elif servico_fornecido == "Gás, água, energia elétrica": L_o, L_f = 0.001, 0.1
    else: return 0, 0
    
    # Norma usa n_z/n_t, mas n_z = número de usuários do serviço, n_t=total de usuários.
    # Assumimos que as pessoas na zona são os usuários para simplificação.
    f_o_serv = n_z / n_t if n_t > 0 else 0
    D2 = r_p * r_f * L_f * f_o_serv
    D3 = L_o * f_o_serv
    return D2, D3

def TABLE_5_9_L3(c_z, c_t, r_p, r_f):
    if c_t == 0: return 0
    # D2 é o único componente de perda para R3
    D2 = r_p * r_f * 0.1 * (c_z / c_t)
    return D2

def calculate_L4(r_p, r_f, le_tipo, lo_tipo, ca, cb, cc, cs):
    c_t = cb + cc + cs
    if c_t == 0: c_t = 1
    Lf_econ = TABLE_5_12_LF_Lo_L4["Agrícola (animais)"] * (ca / (c_t + ca) if (c_t + ca) > 0 else 0)
    Le_econ = r_p * r_f * le_tipo * ((cb + cc) / c_t)
    Lo_econ = lo_tipo * (cs / c_t)
    return {'Lf_econ': Lf_econ, 'Le_econ': Le_econ, 'Lo_econ': Lo_econ}

# ==============================================================================
# SEÇÃO 2: LÓGICA DE AVALIAÇÃO E ANÁLISE (LÓGICA CORRIGIDA E EXPANDIDA)
# ==============================================================================

def run_full_analysis(*args):
    # CORREÇÃO CRÍTICA: A ordem das chaves deve corresponder EXATAMENTE à ordem
    # em que os componentes são adicionados à 'input_list' na interface.
    keys = [
    "L", "W", "H","Hmax", "Ng", "localizacao_cd", "servico", "is_cultural_patrimony",
    "tipo_linha_ct", "roteamento_linha", "ambiente_linha", "tipo_linha_pli",
    "medidas_toque_passo", "tipo_blindagem_linha_cld", "cabeamento_interno",
    "medidas_choque_linha", "DPS_class", "tensao_suportavel_uw",
    "Blindagem_do_Cabo_para_PLD", "resistencia_solo_rs", "Tipo_de_solo",
    "Medidas Contra Incêndio (rp)", "risco_incendio", "Perigo Especial (Pânico, hz)", "pessoas_na_zona",
    "pessoas_total", "Tempo na Zona (h/ano)", # Esta chave é o 26º elemento (índice 25), correspondendo ao input da UI
    "tipo_estrutura_lf_l1", "tipo_estrutura_lo_l1", "tipo_estrutura_le_l4",
    "tipo_estrutura_lo_l4", "Custo de Animais (R$)", "Custo do Edifício (R$)", "Custo do Conteúdo (R$)",
    "Custo dos Sistemas (R$)", "Valor do Patrimônio Cultural (R$)" # Esta é agora a última chave, 35º elemento (índice 34)
]
    params = dict(zip(keys, args))

    # --- Cálculos dos Fatores de Ocorrência (N) ---
    cd_val = TABLE_4_1_CD[params['localizacao_cd']]
    ct_val = TABLE_4_2_CT[params['tipo_linha_ct']]
    ci_val = TABLE_4_3_CI[params['roteamento_linha']]
    ce_val = TABLE_4_4_CE[params['ambiente_linha']]
    n_values = {
        'N_D': N_D(params['Ng'], params['L'], params['W'], params['H'], cd_val),
        'N_M': N_M(params['Ng'], 500, params['L'], params['W']),
        'N_L': N_L(params['Ng'], ct_val, ci_val, ce_val),
        'N_DJ': N_DJ(params['Ng'], params['L'], params['W'], params['H'],params['Hmax'], cd_val, ct_val),
        'N_I': N_I(params['Ng'], ci_val, ce_val, ct_val)
    }

    # --- Cálculos dos Fatores de Perda (L) ---
    ra = TABLE_5_4_Rp[params['Medidas Contra Incêndio (rp)']]
    rf = TABLE_5_5_Rf[params['risco_incendio']]
    
    # L1 (Perda de vidas)
    perdas_l1 = TABLE_5_1_L1(
        r_t=TABLE_5_3_RT[params["Tipo_de_solo"]], L_T=TABLE_5_2_L1_LT,
        n_z=params['pessoas_na_zona'], n_t=params['pessoas_total'], t_z=params['Tempo na Zona (h/ano)'],
        L_o=TABLE_5_2_L1_LO[params['tipo_estrutura_lo_l1']], L_f=TABLE_5_2_L1_LF[params['tipo_estrutura_lf_l1']],
        h_z=TABLE_5_6_HZ[params['Perigo Especial (Pânico, hz)']], r_p=ra, r_f=rf
    )
    l1_values = {'Lf_vida': perdas_l1['D1'], 'Lo_dano_fisico_vida': perdas_l1['D2'], 'Lo_falha_sistema_vida': perdas_l1['D3']}

    # L2 (Perda de serviço público)
    l2_dano, l2_falha = TABLE_5_7_L2(params['pessoas_na_zona'], params['pessoas_total'], ra, rf, params['servico'])

    # L3 (Perda de patrimônio cultural)
    l3_dano = TABLE_5_9_L3(params['Valor do Patrimônio Cultural (R$)'], params['Custo do Edifício (R$)'] + params['Custo do Conteúdo (R$)'], ra, rf)

    # L4 (Perda econômica)
    l4_values = calculate_L4(
        r_p=ra, r_f=rf, le_tipo=TABLE_5_12_LF_Lo_L4[params['tipo_estrutura_le_l4']],
        lo_tipo=TABLE_5_12_Lo_L4_D3[params['tipo_estrutura_lo_l4']], ca=params['Custo de Animais (R$)'],
        cb=params['Custo do Edifício (R$)'], cc=params['Custo do Conteúdo (R$)'], cs=params['Custo dos Sistemas (R$)']
    )

    # --- Cálculo do Custo Total para Perda Anual ---
    total_value = params['Custo do Edifício (R$)'] + params['Custo do Conteúdo (R$)'] + params['Custo dos Sistemas (R$)'] + params['Custo de Animais (R$)']
    if params['is_cultural_patrimony']:
        total_value += params['Valor do Patrimônio Cultural (R$)']
    if total_value == 0: total_value = 1

    results_list = []
    classes_to_test = ["-", "IV", "III", "II", "I"]

    for spda_class in classes_to_test:
        # --- Cálculo das Probabilidades de Dano (P) ---
        pspd_val = TABLE_4_7_PSPD[params['DPS_class']]
        peb_val = TABLE_4_11_PEB[params['DPS_class']]
        pb_val = TABLE_4_6_PB[spda_class]
        pta_val = TABLE_4_5_PTA[params['medidas_toque_passo']]
        pli_val = TABLE_4_13_PLI(params['tensao_suportavel_uw'], params['tipo_linha_pli'])
        cli_val, cld_val = TABLE_4_8_CLI_CLD[params['tipo_blindagem_linha_cld']]
        ks3_val = TABLE_4_9_K3[params['cabeamento_interno']]
        ks4_val = Ks4(params['tensao_suportavel_uw'])
        pld_val = TABLE_4_12_PLD(params['tensao_suportavel_uw'], params['Blindagem_do_Cabo_para_PLD'], params['resistencia_solo_rs'])
        ptu_val = TABLE_4_10_PTU[params['medidas_choque_linha']]
        
        P_A = Pa(pta_val, pb_val); P_B = pb_val; P_C = Pc(pspd_val, cld_val)
        P_M = pspd_val * Pms(ks3_val, ks4_val)
        P_U = Pu(ptu_val, peb_val, pld_val, cld_val)
        P_V = Pv(peb_val, pld_val, cld_val)
        P_W = Pw(pspd_val, pld_val, cld_val)
        P_Z = pspd_val * pli_val * cli_val
        
        # Componentes de Risco
        Ra = n_values['N_D'] * P_A
        Rb = n_values['N_D'] * P_B
        Rc = n_values['N_D'] * P_C
        Rm = n_values['N_M'] * P_M
        Ru = (n_values['N_L'] + n_values['N_DJ']) * P_U
        Rv = (n_values['N_L'] + n_values['N_DJ']) * P_V
        Rw = (n_values['N_L'] + n_values['N_DJ']) * P_W
        Rz = n_values['N_I'] * P_Z

        # --- Lógica de R1 (Risco de Vida) ---
        R1_A = Ra * l1_values['Lf_vida']; R1_B = Rb * l1_values['Lo_dano_fisico_vida']; R1_C = Rc * l1_values['Lo_falha_sistema_vida']
        R1_M = Rm * l1_values['Lo_falha_sistema_vida']; R1_U = Ru * l1_values['Lf_vida']; R1_V = Rv * l1_values['Lo_dano_fisico_vida']
        R1_W = Rw * l1_values['Lo_falha_sistema_vida']; R1_Z = Rz * l1_values['Lo_falha_sistema_vida']
        
        R1_incendio =  R1_C + R1_M + R1_W + R1_Z
        R1_choque = R1_A + R1_B + R1_U + R1_V 
        R1 = R1_choque
        if l1_values['Lo_dano_fisico_vida']>=.01 or l1_values['Lf_vida']>=.05: # Adiciona componentes de incêndio se houver risco
             R1 += R1_incendio

        # --- Lógica de R2 (Serviço Público) ---
        R2_B = Rb * l2_dano; R2_C = Rc * l2_falha; R2_M = Rm * l2_falha
        R2_V = Rv * l2_dano; R2_W = Rw * l2_falha; R2_Z = Rz * l2_falha
        R2 = R2_B + R2_C + R2_M + R2_V + R2_W + R2_Z

        # --- Lógica de R3 (Patrimônio Cultural) ---
        R3_B = Rb * l3_dano; R3_V = Rv * l3_dano
        R3 = R3_B + R3_V

        # --- Lógica de R4 (Risco Econômico) ---
        R4_A = Ra * l4_values['Lf_econ']; R4_B = Rb * l4_values['Le_econ']; R4_C = Rc * l4_values['Lo_econ']
        R4_M = Rm * l4_values['Lo_econ']; R4_U = Ru * l4_values['Lf_econ']; R4_V = Rv * l4_values['Le_econ']
        R4_W = Rw * l4_values['Lo_econ']; R4_Z = Rz * l4_values['Lo_econ']
        
        R4_danos_e_falhas = R4_B + R4_C + R4_M + R4_V + R4_W + R4_Z
        R4 = R4_danos_e_falhas
        if params['Custo de Animais (R$)'] > 0:
            R4 += R4_A + R4_U

        # --- Avaliação e Formatação dos Resultados ---
        RT1, RT2, RT3, RT4 = 1.0e-5, 1.0e-3, 1.0e-4, 1.0e-3
        
        result_row = {"Classe SPDA": "Nenhuma" if spda_class == "-" else spda_class}
        
        # R1 (sempre calculado)
        result_row['R1 (Vida)'] = f"{R1:.2e}"; result_row['Status R1'] = "✅ ACEITÁVEL" if R1 <= RT1 else "❌ INACEITÁVEL"
        
        # R2 (condicional)
        if params['servico'] != '-':
            result_row['R2 (Serviço)'] = f"{R2:.2e}"; result_row['Status R2'] = "✅ ACEITÁVEL" if R2 <= RT2 else "❌ INACEITÁVEL"
        else:
            result_row['R2 (Serviço)'] = 'N/A'; result_row['Status R2'] = 'N/A'
            
        # R3 (condicional)
        if params['is_cultural_patrimony']:
            result_row['R3 (Cultural)'] = f"{R3:.2e}"; result_row['Status R3'] = "✅ ACEITÁVEL" if R3 <= RT3 else "❌ INACEITÁVEL"
        else:
            result_row['R3 (Cultural)'] = 'N/A'; result_row['Status R3'] = 'N/A'
            
        # R4 (sempre calculado)
        custo_anual = R4 * total_value
        result_row['R4 (Econômico)'] = f"{R4:.2e}"; result_row['Status R4'] = "✅ ACEITÁVEL" if R4 <= RT4 else "❌ INACEITÁVEL"
        result_row['Perda Anual (R$)'] = f"{custo_anual:,.2f}"
        
        results_list.append(result_row)
    
    df = pd.DataFrame(results_list)

    # --- Lógica de Recomendação Refinada ---
    active_status_cols = ['Status R1', 'Status R4']
    if params['servico'] != '-': active_status_cols.append('Status R2')
    if params['is_cultural_patrimony']: active_status_cols.append('Status R3')
    
    aceitos = df.copy()
    for status_col in active_status_cols:
        aceitos = aceitos[aceitos[status_col] == "✅ ACEITÁVEL"]

    if not aceitos.empty:
        recomendado = aceitos.iloc[0]
        rec_classe = recomendado['Classe SPDA']
        recomendacao_texto = (f"**Recomendação:** A **Classe SPDA {rec_classe}** é a solução de melhor custo-benefício, pois é a menos rigorosa que atende a todos os critérios de segurança aplicáveis.\n\n"
                              f"Para esta classe, todos os riscos avaliados ({', '.join(col.replace('Status ', '') for col in active_status_cols)}) estão em níveis aceitáveis.")
    else:
        recomendacao_texto = "**Recomendação:** Nenhuma das classes de SPDA padrão foi suficiente para mitigar os riscos. São necessárias medidas de proteção adicionais ou uma reavaliação dos parâmetros de entrada."
        
    # Esconde colunas não utilizadas para uma UI mais limpa
    cols_to_drop = [col for col in ['R2 (Serviço)', 'Status R2', 'R3 (Cultural)', 'Status R3'] if df[col].iloc[0] == 'N/A']
    df = df.drop(columns=cols_to_drop)

    return df, recomendacao_texto


# ==============================================================================
# SEÇÃO 3: DEFINIÇÃO DA INTERFACE GRÁFICA (UI) COM GRADIO
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="Análise de Risco SPDA - NBR 5419") as iface:
    gr.Markdown("# ⚡ Análise de Risco SPDA - NBR 5419")
    gr.Markdown("Esta ferramenta implementa o gerenciamento de risco completo da norma (R1, R2, R3 e R4). Preencha todos os campos para obter a análise detalhada e a recomendação.")

    with gr.Row():
        with gr.Column(scale=2):
            input_list = []
            
            with gr.Accordion("1. Parâmetros da Edificação e Riscos Especiais", open=True):
                with gr.Row():
                    input_list.append(gr.Number(label="Comprimento (L) em m", value=24.3))
                    input_list.append(gr.Number(label="Largura (W) em m", value=25.1))
                    input_list.append(gr.Number(label="Altura (H) em m", value=50))
                    input_list.append(gr.Number(label="Altura mais desnível (Hmax) em m", value=56))
                input_list.append(gr.Number(label="Densidade de Descargas (Ng)", value=0.5, info="raios/km²/ano"))
                input_list.append(gr.Dropdown(label="Localização da Estrutura (Fator Cd)", choices=list(TABLE_4_1_CD.keys()), value="Estrutura cercada por objetos mais altos"))
                
                gr.Markdown("#### Selecione os riscos aplicáveis (além de Vida e Econômico):")
                input_list.append(gr.Dropdown(label="Risco de Perda de Serviço Público (R2)", choices=service, value="-"))

                input_list.append(gr.Checkbox(label="Estrutura é um patrimônio cultural (ativa Risco R3)"))

            with gr.Accordion("2. Parâmetros das Linhas Externas", open=False):
                input_list.append(gr.Dropdown(label="Tipo de Linha (p/ Fator Ct)", choices=list(TABLE_4_2_CT.keys()), value="Linha aérea de energia ou sinal"))
                with gr.Row():
                    input_list.append(gr.Dropdown(label="Roteamento da Linha (Fator Ci)", choices=list(TABLE_4_3_CI.keys()), value="Aéreo"))
                    input_list.append(gr.Dropdown(label="Ambiente da Linha (Fator Ce)", choices=list(TABLE_4_4_CE.keys()), value="Urbano com edifícios > 20m"))
                input_list.append(gr.Dropdown(label="Tipo de Linha (p/ Fator PLI)", choices=["Energia", "Sinal"], value="Energia"))

            with gr.Accordion("3. Medidas e Parâmetros de Proteção (P)", open=False):
                input_list.append(gr.Dropdown(label="Medidas Contra Toque/Passo (Pta)", choices=list(TABLE_4_5_PTA.keys()), value="Sem medidas de proteção"))
                input_list.append(gr.Dropdown(label="Blindagem da Linha Externa (Cld)", choices=list(TABLE_4_8_CLI_CLD.keys()), value="Aérea/Enterrada sem blindagem"))
                input_list.append(gr.Dropdown(label="Cabeamento Interno (Ks3)", choices=list(TABLE_4_9_K3.keys()), value="Cabo sem blindagem evitando grandes laços"))
                input_list.append(gr.Dropdown(label="Medidas Contra Choque na Linha (Ptu)", choices=list(TABLE_4_10_PTU.keys()), value="Isolação elétrica adequada"))
                input_list.append(gr.Dropdown(label="Classe do DPS (p/ Pspd e Peb)", choices=list(TABLE_4_11_PEB.keys()), value="II"))
                with gr.Row():
                    input_list.append(gr.Dropdown(label="Tensão Suportável (Uw)", choices=[1.0, 1.5, 2.5, 4.0, 6.0], value=1.5))
                    input_list.append(gr.Dropdown(label="Blindagem do Cabo (p/ PLD)", choices=["nenhum ou não interligada", "blindado e interligada"], value="nenhum ou não interligada"))
                    input_list.append(gr.Slider(label="Resistência do Solo (Rs)", minimum=1, maximum=20, value=1, step=1))

            with gr.Accordion("4. Parâmetros de Perda (L)", open=False):
                input_list.append(gr.Dropdown(label="Tipo de Piso (rt)", choices=list(TABLE_5_3_RT.keys()), value="Concreto/agricultura (resistividade <= 1 kΩ.m)"))
                input_list.append(gr.Dropdown(label="Medidas Contra Incêndio (rp)", choices=list(TABLE_5_4_Rp.keys()), value="Sem medidas"))
                input_list.append(gr.Dropdown(label="risco_incendio", choices=list(TABLE_5_5_Rf.keys()), value="Risco ordinário"))
                input_list.append(gr.Dropdown(label="Perigo Especial (Pânico, hz)", choices=list(TABLE_5_6_HZ.keys()), value="Médio pânico/Dificuldade de evacuação (100-1000 pessoas)"))
                with gr.Row():
                    input_list.append(gr.Number(label="Pessoas na Zona / Usuários do Serviço", value=112))
                    input_list.append(gr.Number(label="Pessoas Total / Total de Usuários", value=112))
                    input_list.append(gr.Number(label="Tempo na Zona (h/ano)", value=4380))
                gr.Markdown("##### Fatores de Perda (L) por Tipo de Estrutura")
                with gr.Row():
                    input_list.append(gr.Dropdown(label="L1-D2 (Vida/Dano Físico)", choices=list(TABLE_5_2_L1_LF.keys()), value="Outros"))
                    input_list.append(gr.Dropdown(label="L1-D3 (Vida/Falha Sistema)", choices=list(TABLE_5_2_L1_LO.keys()), value="Aplicações gerais"))
                with gr.Row():
                    input_list.append(gr.Dropdown(label="L4-D2 (Econ/Dano Físico)", choices=list(TABLE_5_12_LF_Lo_L4.keys()), value="Outros"))
                    input_list.append(gr.Dropdown(label="L4-D3 (Econ/Falha Sistema)", choices=list(TABLE_5_12_Lo_L4_D3.keys()), value="Outros"))

            with gr.Accordion("5. Análise Financeira", open=True):
                gr.Markdown("Insira os custos totais para a análise econômica (R4) e de patrimônio (R3).")
                with gr.Row():
                    input_list.append(gr.Number(label="Custo de Animais (R$)", value=0))
                    input_list.append(gr.Number(label="Custo do Edifício (R$)", value=30000000))
                with gr.Row():
                    input_list.append(gr.Number(label="Custo do Conteúdo (R$)", value=5000000))
                    input_list.append(gr.Number(label="Custo dos Sistemas (R$)", value=800000))
                input_list.append(gr.Number(label="Valor do Patrimônio Cultural (R$)", value=0, info="Preencha se o Risco R3 foi ativado acima."))
            
            submit_button = gr.Button("Analisar Riscos e Recomendar Proteção", variant="primary")

        with gr.Column(scale=3):
            gr.Markdown("## Resultados da Análise")
            # CORREÇÃO: Headers e datatypes são removidos para permitir que o backend defina as colunas dinamicamente.
            output_df = gr.Dataframe(label="Tabela de Riscos por Classe de Proteção", wrap=True)
            output_recommendation = gr.Markdown(label="Recomendação Final")

    # Ação do botão
    submit_button.click(fn=run_full_analysis, inputs=input_list, outputs=[output_df, output_recommendation])

if __name__ == "__main__":
    iface.launch()
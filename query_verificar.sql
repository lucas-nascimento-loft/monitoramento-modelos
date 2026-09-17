WITH
first_defaults AS (
  SELECT
    contract_id,
    DATE(MIN(pendency_created_at)) AS first_comunicacao_date,
    DATE(MIN(pendency_at)) AS first_competencia_date,
    DATE(MIN(payment_at)) AS first_payment_date,
    DATE(MIN(reimbursement_at)) AS first_reimbursement_date
  FROM `loft-dl-fintech.cp_gold.watchlist_fact`
  WHERE pendency_type IN ('Inadimplência')
  GROUP BY contract_id
),

tb_leads as(
  SELECT
    rf.contract_id,
    date(cf.requested_at) as requested_at,
    cf.tipo,
    cf.tipo_contrato,
    cf.rule,
    cf.bureau_nm,
    cf.modeloBlend,
    rf.approved_products,
    case when '32' in unnest(json_value_array(rf.approved_products)) then 1 else 0 end as flag_pop32,
    rd.product_nm,
    -- case when cf.modeloBlend = 'BLEND_REGRESSAO_2026' and cf.bureau_nm in ('BLEND_REGRESSAO_2026', 'BVS_CUSTOM', 'HVA3') then 'BLEND2'
    --     when cf.modeloBlend in ('BLEND3_3', 'BLEND3_4') and cf.bureau_nm in ('BLEND3_3', 'BLEND3_4', 'BVS_CUSTOM', 'HVA4') then 'BLEND3'
    --     when cf.modeloBlend in ('BLEND_4', 'BVS_CUSTOM_V2', 'HFT1') and cf.bureau_nm in ('BLEND_4', 'BVS_CUSTOM_V2', 'HFT1') then 'BLEND4'
    --     else 'Outros' end as bureau_nm_ajust,
    -- case when cf.modeloBlend in ('BLEND3_3', 'BLEND3_4') and cf.bureau_nm in ('BVS_CUSTOM', 'HVA4') then 1
    --     when cf.bureau_nm in ('BVS_CUSTOM_V2', 'HFT1') then 1
    --     else 0 end as is_fallback, -- apenas blend3 e blend4
    case when cf.modeloBlend in ('BLEND_REGRESSAO_2026', 'BLEND_REGRESSAO') then 'BLEND2'
         when cf.modeloBlend in ('BLEND3_2', 'BLEND3_3', 'BLEND3_4') then 'BLEND3'
         when cf.modeloBlend in ('BLEND_4', 'BVS_CUSTOM_V2', 'HFT1') then 'BLEND4'
         when cf.modeloBlend in ('BLEND_4', 'BVS_CUSTOM_V2', 'HFT1') then 'BLEND4'
         when coalesce(cf.modeloBlend, '') = '' and cf.bureau_nm in ('BLEND_REGRESSAO', 'BLEND_REGRESSAO_OTIMISTA', 'BVS_CUSTOM', 'BVS_CUSTOM_OTIMISTA') then 'BLEND2'
         else 'OUTROS' end as bureau_nm_ajust,
    i.id_imobiliaria,
    i.id_cidade_ibge,
    i.cidade as imovel_cidade,
    i.uf as imovel_uf,
    ad.segmentacao as agency_segmentacao,
    rd.lead_elegivel,
    rd.proposta_iniciada,
    rd.proposta_enviada,
    rd.proposta_aprovada,
    rd.proposta_ativada,
    rd.is_activeted,
    rf.activated_at,
    rd.aprovado_motor_mesa,
    coalesce(cf.rating_score_ds, '-1') as rating_score_ds,
    case when coalesce(cf.qtd_proponentes, -1) <= 1 then 1 else 0 end as is_single_proponent,
    ca.blend_regressao_predict_nr,

    COALESCE(rd.is_activeted, cf.is_activeted) AS is_activeted,

    DATE_DIFF(
      DATE(CURRENT_DATE()),
      COALESCE(DATE(rf.activated_at), DATE(cf.requested_at)),
      DAY
    ) AS time2requested,

    CASE
      WHEN fd.first_competencia_date IS NULL OR rf.activated_at IS NULL THEN NULL
      ELSE DATE_DIFF(
        DATE_TRUNC(DATE(fd.first_competencia_date), MONTH),
        DATE_TRUNC(DATE(rf.activated_at), MONTH),
        MONTH
      )
    END AS time2def_pc,

    CASE
      WHEN fd.first_comunicacao_date IS NULL OR rf.activated_at IS NULL THEN NULL
      ELSE DATE_DIFF(
        DATE_TRUNC(DATE(fd.first_comunicacao_date), MONTH),
        DATE_TRUNC(DATE(rf.activated_at), MONTH),
        MONTH
      )
    END AS time2def_pcc,

    CASE
      WHEN fd.first_payment_date IS NULL OR rf.activated_at IS NULL THEN NULL
      ELSE DATE_DIFF(
        DATE_TRUNC(DATE(fd.first_payment_date), MONTH),
        DATE_TRUNC(DATE(rf.activated_at), MONTH),
        MONTH
      )
    END AS time2def_pd,

    CASE
      WHEN fd.first_reimbursement_date IS NULL OR rf.activated_at IS NULL THEN NULL
      ELSE DATE_DIFF(
        DATE_TRUNC(DATE(fd.first_reimbursement_date), MONTH),
        DATE_TRUNC(DATE(rf.activated_at), MONTH),
        MONTH
      )
    END AS time2def_reimbursement,

    fd.first_comunicacao_date,
    fd.first_competencia_date,
    fd.first_payment_date,
    fd.first_reimbursement_date

  from loft-dl-fintech.cp_gold.requests_fact as rf
  left join loft-dl-fintech.cp_gold.requests_dim AS rd
    on rf.contract_id = rd.contract_id
  left join loft-dl-fintech.cp_gold.credit_fact as cf
    on rf.contract_id = cf.contract_id
  left join loft-dl-fintech.cp_silver.int_credit_analyses as ca
    on rf.contract_id = ca.contract_id
  left join loft-dl-fintech.bronze_credpago_enriched.imovel as i
    on rf.contract_id = i.id
  left join loft-dl-fintech.cp_gold.agency_dim as ad
    on rd.agency_id = ad.agency_id
  left join first_defaults as fd
    on rf.contract_id = fd.contract_id
  where cf.tipo_contrato = 'PF' and
    date(cf.requested_at) >= date('2024-01-01') and
    date(cf.requested_at) < date(current_date())
),

leads_e_defaults_mob AS (
  SELECT
    *,
-- PC
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_competencia_date IS NOT NULL AND time2requested >= 120 AND time2def_pc <= 2 THEN 1
      WHEN time2requested >= 120 THEN 0
      ELSE NULL
    END AS pc_2m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_competencia_date IS NOT NULL AND time2requested >= 150 AND time2def_pc <= 3 THEN 1
      WHEN time2requested >= 150 THEN 0
      ELSE NULL
    END AS pc_3m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_competencia_date IS NOT NULL AND time2requested >= 180 AND time2def_pc <= 4 THEN 1
      WHEN time2requested >= 180 THEN 0
      ELSE NULL
    END AS pc_4m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_competencia_date IS NOT NULL AND time2requested >= 240 AND time2def_pc <= 6 THEN 1
      WHEN time2requested >= 240 THEN 0
      ELSE NULL
    END AS pc_6m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_competencia_date IS NOT NULL AND time2requested >= 330 AND time2def_pc <= 9 THEN 1
      WHEN time2requested >= 330 THEN 0
      ELSE NULL
    END AS pc_9m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_competencia_date IS NOT NULL AND time2requested >= 420 AND time2def_pc <= 12 THEN 1
      WHEN time2requested >= 420 THEN 0
      ELSE NULL
    END AS pc_12m,

-- PCC
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_comunicacao_date IS NOT NULL AND time2requested >= 60 AND time2def_pcc <= 2 THEN 1
      WHEN time2requested >= 60 THEN 0
      ELSE NULL
    END AS pcc_2m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_comunicacao_date IS NOT NULL AND time2requested >= 90 AND time2def_pcc <= 3 THEN 1
      WHEN time2requested >= 90 THEN 0
      ELSE NULL
    END AS pcc_3m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_comunicacao_date IS NOT NULL AND time2requested >= 120 AND time2def_pcc <= 4 THEN 1
      WHEN time2requested >= 120 THEN 0
      ELSE NULL
    END AS pcc_4m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_comunicacao_date IS NOT NULL AND time2requested >= 180 AND time2def_pcc <= 6 THEN 1
      WHEN time2requested >= 180 THEN 0
      ELSE NULL
    END AS pcc_6m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_comunicacao_date IS NOT NULL AND time2requested >= 270 AND time2def_pcc <= 9 THEN 1
      WHEN time2requested >= 270 THEN 0
      ELSE NULL
    END AS pcc_9m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_comunicacao_date IS NOT NULL AND time2requested >= 360 AND time2def_pcc <= 12 THEN 1
      WHEN time2requested >= 360 THEN 0
      ELSE NULL
    END AS pcc_12m,

-- PD
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_payment_date IS NOT NULL AND time2requested >= 60 AND time2def_pd <= 2 THEN 1
      WHEN time2requested >= 60 THEN 0
      ELSE NULL
    END AS pd_2m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_payment_date IS NOT NULL AND time2requested >= 90 AND time2def_pd <= 3 THEN 1
      WHEN time2requested >= 90 THEN 0
      ELSE NULL
    END AS pd_3m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_payment_date IS NOT NULL AND time2requested >= 120 AND time2def_pd <= 4 THEN 1
      WHEN time2requested >= 120 THEN 0
      ELSE NULL
    END AS pd_4m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_payment_date IS NOT NULL AND time2requested >= 150 AND time2def_pd <= 5 THEN 1
      WHEN time2requested >= 150 THEN 0
      ELSE NULL
    END AS pd_5m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_payment_date IS NOT NULL AND time2requested >= 180 AND time2def_pd <= 6 THEN 1
      WHEN time2requested >= 180 THEN 0
      ELSE NULL
    END AS pd_6m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_payment_date IS NOT NULL AND time2requested >= 270 AND time2def_pd <= 9 THEN 1
      WHEN time2requested >= 270 THEN 0
      ELSE NULL
    END AS pd_9m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_payment_date IS NOT NULL AND time2requested >= 360 AND time2def_pd <= 12 THEN 1
      WHEN time2requested >= 360 THEN 0
      ELSE NULL
    END AS pd_12m,

-- Reimbursement
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_reimbursement_date IS NOT NULL AND time2requested >= 60 AND time2def_reimbursement <= 2 THEN 1
      WHEN time2requested >= 60 THEN 0
      ELSE NULL
    END AS reimbursement_2m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_reimbursement_date IS NOT NULL AND time2requested >= 90 AND time2def_reimbursement <= 3 THEN 1
      WHEN time2requested >= 90 THEN 0
      ELSE NULL
    END AS reimbursement_3m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_reimbursement_date IS NOT NULL AND time2requested >= 120 AND time2def_reimbursement <= 4 THEN 1
      WHEN time2requested >= 120 THEN 0
      ELSE NULL
    END AS reimbursement_4m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_reimbursement_date IS NOT NULL AND time2requested >= 150 AND time2def_reimbursement <= 5 THEN 1
      WHEN time2requested >= 150 THEN 0
      ELSE NULL
    END AS reimbursement_5m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_reimbursement_date IS NOT NULL AND time2requested >= 180 AND time2def_reimbursement <= 6 THEN 1
      WHEN time2requested >= 180 THEN 0
      ELSE NULL
    END AS reimbursement_6m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_reimbursement_date IS NOT NULL AND time2requested >= 270 AND time2def_reimbursement <= 9 THEN 1
      WHEN time2requested >= 270 THEN 0
      ELSE NULL
    END AS reimbursement_9m,
    CASE
      WHEN activated_at IS NULL THEN NULL
      WHEN first_reimbursement_date IS NOT NULL AND time2requested >= 360 AND time2def_reimbursement <= 12 THEN 1
      WHEN time2requested >= 360 THEN 0
      ELSE NULL
    END AS reimbursement_12m

    FROM tb_leads
)

SELECT
  date_trunc(COALESCE(DATE(activated_at), DATE(requested_at)), MONTH) as safra,
  avg(pc_3m) as pc_3m,
  avg(pcc_3m) as pcc_3m,
  avg(pd_3m) as pd_3m,
  avg(reimbursement_3m) as reimbursement_3m,
  avg(pcc_6m) as pcc_6m,
  avg(reimbursement_6m) as reimbursement_6m
FROM leads_e_defaults_mob
WHERE date_trunc(COALESCE(DATE(activated_at), DATE(requested_at)), MONTH) >= date('2025-01-01')
group by 1
order by 1 
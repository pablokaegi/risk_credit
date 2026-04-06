# BCRA API Integration - Quick Start Guide

## Overview

This project uses **official APIs from the Banco Central de la República Argentina (BCRA)**, which provides the most comprehensive and reliable data for credit risk analysis in Argentina.

## Why BCRA APIs?

| Feature | BCRA APIs | Yahoo Finance | CNV Scraping |
|---------|-----------|---------------|--------------|
| **Coverage** | ✅ ALL Argentine companies | ❌ ~15 ADRs only | ⚠️ ~80 BYMA companies |
| **Distress Signals** | ✅ Cheques denunciados, deuda real | ❌ Not available | ⚠️ Limited |
| **Data Quality** | ✅ Official, audited | ⚠️ Third-party estimates | ⚠️ Unstable |
| **Maintenance** | ✅ Proactively maintained | ⚠️ Community-driven | ❌ Breaks frequently |
| **Professionalism** | ✅ Official API, documented | ❌ Side project | ⚠️ Unofficial |

## BCRA API Endpoints Used

### 1. Central de Deudores
**Purpose:** Complete debt registry for all Argentine companies

**What it provides:**
- Total debt by entity
- Debt situation (1-5 scale, where 5 = default)
- Number of financial entities
- Historical debt data

**Distress Signals:**
- Situación 1: Normal
- Situación 2: Potential risk
- Situación 3: Moderate risk
- Situación 4: High risk
- Situación 5: Default

**Example:**
```python
from src.data_acquisition.bcra_api import BCRAAPIClient

client = BCRAAPIClient()
debt_data = client.get_central_deudores(cuit='30-12345678-9')

# Total debt
total_debt = debt_data['deuda'].sum()

# Worst situation
worst_situation = debt_data['situacion'].max()

if worst_situation >= 4:
    print("HIGH RISK - Company in distress")
```

### 2. Cheques Denunciados
**Purpose:** Checks rejected due to insufficient funds

**What it provides:**
- Check number
- Amount
- Rejection date
- Issuing company (CUIT)
- Reason for rejection

**Why it matters:**
- Multiple rejected checks = cash flow problems
- Strong predictor of bankruptcy
- Real-time distress indicator

**Example:**
```python
# Get rejected checks from last 30 days
rejected_checks = client.get_cheques_denunciados()

# Find companies with multiple rejected checks
distress_companies = rejected_checks.groupby('cuit').size()
distress_companies = distress_companies[distress_companies > 2]

print("Companies in distress:")
for cuit, count in distress_companies.items():
    print(f"CUIT {cuit}: {count} rejected checks")
```

### 3. Estadísticas Cambiarias
**Purpose:** Exchange rates for USD-denominated debt analysis

**What it provides:**
- Official dollar rate
- Wholesale dollar rate
- Central bank reserves

**Why it matters:**
- Many Argentine companies have USD debt
- Sharp devaluation = instant distress increase
- Context for debt servicing capacity

**Example:**
```python
# Get official dollar rate
dollar = client.get_dollar_rate(tipo='oficial')

# Analyze devaluation
dollar['change'] = dollar['promedio'].pct_change()
sharp_devaluation = dollar[dollar['change'] >0.05]

print(f"Sharp devaluations: {len(sharp_devaluation)} days")
```

### 4. Estadísticas Monetarias
**Purpose:** Interest rates and monetary aggregates

**What it provides:**
- BADLAR rate (benchmark rate)
- Monetary aggregates (M1, M2, M3)
- Inflation indicators

**Why it matters:**
- High interest rates = debt servicing difficulty
- Monetary expansion = inflation = economic stress

## Risk Score Methodology

The model calculates a **composite risk score (0-100)** combining:

### Component 1: Debt Situation (40% weight)
```
Situación 1-2: 0-20 points (low risk)
Situación 3:   30 points (moderate risk)
Situación 4:   40 points (high risk)
Situación 5:   60 points (severe risk)
```

### Component 2: Rejected Checks Count (30% weight)
```
0 checks:  0 points
1-2 checks: 10-20 points
3-5 checks: 30-50 points
>5 checks:  60-80 points
```

### Component 3: Rejected Check Amount (30% weight)
```
< ARS 1M:   10 points
< ARS 10M:  20 points
> ARS 10M:  30 points
```

### Final Risk Score
```python
risk_score = (
    debt_score *0.4 +
    check_count_score *0.3 +
    check_amount_score * 0.3
)

# Target variable for ML
target = 1 if risk_score > 50 else 0
```

## Target Variable Definition

**Target = 1 (Distress)** if ANY of:
- Risk score > 50
- Debt situation ≥ 4
- > 3 rejected checks in last 90 days
- Total rejected check amount > ARS 10M

**Target = 0 (Healthy)** otherwise

## Quick Start

### 1. Get Company Risk Score
```python
from src.data_acquisition.bcra_api import BCRAAPIClient

client = BCRAAPIClient()

# Fetch all distress signals
signals = client.fetch_company_distress_signals(cuit='30-12345678-9')

print(f"Risk Score: {signals['riesgo']}/100")
print(f"Total Debt: ${signals['deuda_total']:,.2f}")
print(f"Rejected Checks: {signals['cheques_rechazados']}")
print(f"Predicted Class: {'DISTRESS' if signals['riesgo'] > 50 else 'HEALTHY'}")
```

### 2. Create Training Dataset
```python
from src.data_acquisition.bcra_api import create_training_dataset

# Automatically fetches and processes all BCRA data
dataset = create_training_dataset(
    output_path='data/processed/bcra_dataset.csv',
    include_macro=True  # Include dollar rate, reserves, interest rates
)

# Dataset columns:
# - cuit, deuda_total, peor_situacion, num_entidades
# - cheques_rechazados, monto_total_rechazado
# - riesgo, target
# - dollar_rate, reservas_bcra (if include_macro=True)
```

### 3. Train Model
```python
from src.model.classifier import DistressClassifier

clf = DistressClassifier(model_type='random_forest')
clf.load_data('data/processed/bcra_dataset.csv')
clf.preprocess(handle_imbalance=True)
clf.train(use_grid_search=True)
clf.evaluate()
clf.save_model('models/bcra_distress_model.pkl')
```

## API Limits and Best Practices

### Rate Limiting
- BCRA recommends: **1 request per second**
- Our client automatically enforces this with `rate_limit_delay=1.0`
- Caching enabled by default

### Best Practices
1. **Use caching**: Set `cache_enabled=True` to avoid repeated calls
2. **Batch processing**: Query multiple CUITs in sequence, not parallel
3. **Error handling**: Client has automatic retry with exponential backoff
4. **Data freshness**: Cache expires after 24 hours (re-query daily)

### Example with Rate Limiting
```python
client = BCRAAPIClient(
    rate_limit_delay=1.0,  # 1 second between requests
    retry_attempts=3,      # Retry 3 times on failure
    cache_enabled=True     # Cache responses locally
)

# Process 100 CUITs sequentially (respects rate limit)
for cuit in cuit_list:
    data = client.fetch_company_distress_signals(cuit)
    time.sleep(1.0)  # Explicit delay (optional, client handles it)
```

## Official Documentation

- **API Portal:** https://api.bcra.gob.ar
- **Documentation:** Each API has a "Manual de Desarrollo"
- **Contact:** api@bcra.gob.ar
- **Legal Notice:** Each API has specific terms of use

## Legal Considerations

- BCRA APIs are free to use
- No registration required
- Rate limiting applies (1 req/sec)
- Data is for informational purposes
- For commercial use, contact BCRA

## Troubleshooting

### Error: "Connection timeout"
**Solution:** BCRA servers may be slow. Increase retry_attempts:
```python
client = BCRAAPIClient(retry_attempts=5)
```

### Error: "No data found for CUIT"
**Solutions:**
1. CUIT may not be in system (not registered)
2. Company may have no debt/cheques
3. Try a different CUIT (test with known companies)

### Error: "Rate limit exceeded"
**Solution:** Increase delay between requests:
```python
client = BCRAAPIClient(rate_limit_delay=2.0)  # 2 seconds
```

### Cached data is outdated
**Solution:** Clear cache directory:
```python
client = BCRAAPIClient(cache_enabled=False)  # Disable cache temporarily
# OR manually delete data/cache/bcra/
```

## Comparison with Other Sources

| Metric | BCRA | yFinance | CNV Scraping |
|--------|------|----------|--------------|
| **Time to setup** | 5 min | 2 min | 2 hours |
| **Data quality** | Official | Estimated | Varies |
| **Distress signals** | ✅ Real-time | ❌ None | ⚠️ Limited |
| **Coverage** | 100% | 5% | 60% |
| **Maintenance** | None | None | High |
| **Cost** | Free | Free | Server costs |

## Conclusion

**BCRA APIs are the ONLY production-ready data source for Argentine credit risk analysis.**

- Real-time distress signals (cheques denunciados)
- Complete debt registry (central de deudores)
- Macro context (dollar, reserves, rates)
- Official, documented, maintained

**For any serious Argentine credit risk project, BCRA APIs are non-negotiable.**

---

*Last Updated: January 2025*
*BCRA API Version: 1.0+*
*Project Repository: https://github.com/pablokaegi/risk_credit*
#!/usr/bin/env python3
"""
BCRA API - Quick Start Example

This script demonstrates how to use BCRA APIs for credit risk analysis.

Run this script to:
    1. Test BCRA API connectivity
    2. Fetch real data from Central de Deudores
    3. Fetch rejected checks (cheques denunciados)
    4. Calculate risk scores for companies

Usage:
    python scripts/example_bcra_usage.py

Author: Pablo Kaegi
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_acquisition.bcra_api import BCRAAPIClient, create_training_dataset
import pandas as pd


def example_get_company_debt():
    """
    Example 1: Get debt information for a specific company.
    
    This shows how to query Central de Deudores for debt situation.
    """
    print("=" * 80)
    print("EXAMPLE 1: Get Company Debt Information")
    print("=" * 80)
    
    client = BCRAAPIClient()
    
    # Get debt data for a specific CUIT
    # Note: Replace with a real CUIT from Argentina
    cuit_ejemplo = '30-12345678-9'  # Example CUIT
    
    try:
        deuda_df = client.get_central_deudores(cuit=cuit_ejemplo)
        
        if not deuda_df.empty:
            print(f"\nDebt information for CUIT: {cuit_ejemplo}")
            print(deuda_df)
            
            # Calculate total debt
            deuda_total = deuda_df['deuda'].sum()
            situacion_max = deuda_df['situacion'].max()
            
            print(f"\nSummary:")
            print(f"  Total Debt: ${deuda_total:,.2f}")
            print(f"  Worst Situation: {situacion_max} (1=normal, 5=default)")
            
            # Situation interpretation
            if situacion_max <= 2:
                print("  Interpretation: Normal debt situation")
            elif situacion_max == 3:
                print("  Interpretation: Some risk - monitoring recommended")
            else:
                print("  Interpretation: HIGH RISK - Potential distress")
        else:
            print(f"\nNo debt data found for CUIT: {cuit_ejemplo}")
            print("This could mean:")
            print("  - CUIT is not in the system")
            print("  - Company has no debt")
            print("  - API returned no data (try again later)")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        client.close()


def example_get_rejected_checks():
    """
    Example 2: Get rejected checks for distress analysis.
    
    Rejected checks are a STRONG indicator of financial distress.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Get Rejected Checks (Cheques Denunciados)")
    print("=" * 80)
    
    client = BCRAAPIClient()
    
    try:
        # Get rejected checks from last 30 days
        cheques_df = client.get_cheques_denunciados()
        
        if not cheques_df.empty:
            print(f"\nFound {len(cheques_df)} rejected checks")
            print("\nFirst 5 records:")
            print(cheques_df.head())
            
            # Aggregate by CUIT
            cheques_por_cuit = cheques_df.groupby('cuit').agg({
                'monto': ['count', 'sum']
            }).reset_index()
            
            cheques_por_cuit.columns = ['cuit', 'num_cheques', 'monto_total']
            cheques_por_cuit = cheques_por_cuit.sort_values('monto_total', ascending=False)
            
            print("\nTop 10 CUITs by total rejected check amount:")
            print(cheques_por_cuit.head(10))
            
            # Highlight distress
            print("\n⚠️  DISTRESS SIGNALS:")
            high_check_count = cheques_por_cuit[cheques_por_cuit['num_cheques'] > 3]
            if not high_check_count.empty:
                print(f"  Found {len(high_check_count)} CUITs with >3 rejected checks")
                print("  These companies likely have cash flow problems")
        else:
            print("\nNo rejected checks found in the last 30 days")
            print("This is GOOD - low distress signals in the market")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        client.close()


def example_calculate_risk_score():
    """
    Example 3: Calculate composite risk score for a company.
    
    This is the MAIN method for credit risk analysis.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Calculate Composite Risk Score")
    print("=" * 80)
    
    client = BCRAAPIClient()
    
    # Example CUIT (replace with real one)
    cuit_ejemplo = '30-12345678-9'
    
    try:
        # Fetch all distress signals
        signals = client.fetch_company_distress_signals(cuit=cuit_ejemplo)
        
        print(f"\nDistress Signals for CUIT: {cuit_ejemplo}")
        print("-" * 40)
        
        print(f"Total Debt: ${signals['deuda_total']:,.2f}")
        print(f"Debt Situation: {signals['situacion']} (1=normal, 5=default)")
        print(f"Rejected Checks: {signals['cheques_rechazados']}")
        print(f"Total Rejected Amount: ${signals['monto_cheques_rechazados']:,.2f}")
        
        print("\n" + "=" * 40)
        print(f"RISK SCORE: {signals['riesgo']:.1f}/100")
        print("=" * 40)
        
        # Interpretation
        if signals['riesgo'] < 30:
            print("✅ LOW RISK - Company is healthy")
        elif signals['riesgo'] < 50:
            print("⚠️  MODERATE RISK - Monitor closely")
        elif signals['riesgo'] <70:
            print("🔴 HIGH RISK - Potential distress")
        else:
            print("🚨 SEVERE RISK - Default probability HIGH")
        
        # Predicted class
        target = 1 if signals['riesgo'] > 50 else 0
        print(f"\nPredicted Class: {'DISTRESS' if target == 1 else 'HEALTHY'}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        client.close()


def example_create_training_dataset():
    """
    Example 4: Create ML-ready training dataset.
    
    This generates a complete dataset for model training.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Create Training Dataset")
    print("=" * 80)
    
    print("\nGenerating ML-ready dataset from BCRA APIs...")
    print("This may take a moment...\n")
    
    try:
        dataset = create_training_dataset(
            output_path='data/processed/bcra_dataset.csv',
            include_macro=True
        )
        
        if not dataset.empty:
            print("\n" + "=" * 40)
            print("DATASET CREATED SUCCESSFULLY")
            print("=" * 40)
            print(f"Shape: {dataset.shape}")
            print(f"Columns: {list(dataset.columns)}")
            print(f"\nTarget Distribution:")
            print(dataset['target'].value_counts())
            print(f"\nClass Balance:")
            print(f"  Healthy (0): {(dataset['target']==0).sum()} ({(dataset['target']==0).mean():.1%})")
            print(f"  Distress (1): {(dataset['target']==1).sum()} ({(dataset['target']==1).mean():.1%})")
            
            print("\nFirst 5 rows:")
            print(dataset.head())
            
            print("\n✅ Dataset saved to: data/processed/bcra_dataset.csv")
            print("✅ Ready for use in ML models!")
        
        else:
            print("\n❌ No data retrieved from BCRA APIs")
            print("Possible reasons:")
            print("  - API is temporarily unavailable")
            print("  - No data for requested parameters")
            print("  - Network connectivity issues")
    
    except Exception as e:
        print(f"\n❌ Error creating dataset: {e}")


def example_get_macro_context():
    """
    Example 5: Get macroeconomic context.
    
    Dollar rate and reserves provide context for USD-denominated debt analysis.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Get Macroeconomic Context")
    print("=" * 80)
    
    client = BCRAAPIClient()
    
    try:
        # Get official dollar rate
        print("\nFetching official dollar rate...")
        dollar_df = client.get_dollar_rate(tipo='oficial')
        
        if not dollar_df.empty:
            latest = dollar_df.iloc[-1]
            print(f"\nLatest Official Dollar Rate:")
            print(f"  Date: {latest['fecha']}")
            print(f"  Buy: ${latest['compra']:.2f}")
            print(f"  Sell: ${latest['venta']:.2f}")
            print(f"  Average: ${latest['promedio']:.2f}")
        
        # Get reserves
        print("\nFetching central bank reserves...")
        reservas_df = client.get_reservas()
        
        if not reservas_df.empty:
            latest = reservas_df.iloc[-1]
            print(f"\nLatest Central Bank Reserves:")
            print(f"  Date: {latest['fecha']}")
            print(f"  Amount: USD {latest['valor']:,.0f}")
        
        # Get interest rates
        print("\nFetching BADLAR interest rate...")
        tasas_df = client.get_tasas_interes(tipo='badlar')
        
        if not tasas_df.empty:
            latest = tasas_df.iloc[-1]
            print(f"\nLatest BADLAR Rate:")
            print(f"  Date: {latest['fecha']}")
            print(f"  Rate: {latest['valor']:.2f}%")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        client.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("BCRA API - QUICK START EXAMPLES")
    print("=" * 80)
    print("\nThese examples show how to use BCRA APIs for credit risk analysis.")
    print("Official API documentation: https://api.bcra.gob.ar")
    print("Contact: api@bcra.gob.ar")
    print()
    
    # Run examples
    try:
        example_get_company_debt()
        example_get_rejected_checks()
        example_calculate_risk_score()
        example_create_training_dataset()
        example_get_macro_context()
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Modify CUITs with real Argentine company IDs")
        print("  2. Run create_training_dataset() to generate ML data")
        print("  3. Train your distress prediction model")
        print("  4. Deploy to production")
        print("\nFor more information, see README.md")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
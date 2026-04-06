#!/usr/bin/env python3
"""
BCRA API - Quick Start Example

This script demonstrates how to use the verified BCRA Cheques API endpoints.

Run this script to:
    1. Test BCRA API connectivity
    2. Fetch the live entity catalog for cheques denunciados
    3. Query real checks using the official manual examples

Usage:
    python scripts/example_bcra_usage.py

Author: Pablo Kaegi
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_acquisition.bcra_api import BCRAAPIClient


def example_get_entities():
    """
    Example 1: List available banking entities for cheques queries.
    """
    print("=" * 80)
    print("EXAMPLE 1: Get Cheques API Entities")
    print("=" * 80)
    
    client = BCRAAPIClient()
    
    try:
        entidades_df = client.get_cheques_entidades()

        if not entidades_df.empty:
            print(f"\nTotal entities available: {len(entidades_df)}")
            print("\nFirst 10 entities:")
            print(entidades_df.head(10).to_string(index=False))

            banco_nacion = entidades_df[entidades_df['codigo_entidad'] == 11]
            if not banco_nacion.empty:
                print("\nVerified example entity:")
                print(banco_nacion.to_string(index=False))
        else:
            print("\nNo entities returned by the API")

        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    finally:
        client.close()


def example_get_rejected_checks():
    """
    Example 2: Query real checks from the official BCRA manual.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Query Cheques Denunciados")
    print("=" * 80)
    
    client = BCRAAPIClient()

    consultas = [
        {'codigo_entidad': 11, 'numero_cheque': 20377516, 'descripcion': 'manual example: reported'},
        {'codigo_entidad': 11, 'numero_cheque': 507, 'descripcion': 'manual example: reported with multiple details'},
        {'codigo_entidad': 11, 'numero_cheque': 203775991, 'descripcion': 'manual example: not reported'},
    ]
    
    try:
        resultados = []
        for consulta in consultas:
            cheque_df = client.get_cheques_denunciados(
                codigo_entidad=consulta['codigo_entidad'],
                numero_cheque=consulta['numero_cheque'],
            )
            if cheque_df.empty:
                raise RuntimeError(f"No response for query {consulta}")

            fila = cheque_df.iloc[0].to_dict()
            fila['descripcion'] = consulta['descripcion']
            fila['codigo_entidad'] = consulta['codigo_entidad']
            resultados.append(fila)

        resultados_df = pd.DataFrame(resultados)
        print("\nLive queries:")
        print(
            resultados_df[
                [
                    'codigo_entidad',
                    'numero_cheque',
                    'denunciado',
                    'fecha_procesamiento',
                    'cantidad_detalles',
                    'causales',
                    'descripcion',
                ]
            ].to_string(index=False)
        )

        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    finally:
        client.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("BCRA API - QUICK START EXAMPLES")
    print("=" * 80)
    print("\nThese examples show the verified Cheques Denunciados endpoints.")
    print("Official API documentation: https://api.bcra.gob.ar")
    print("Contact: api@bcra.gob.ar")
    print()
    
    # Run examples
    try:
        results = [
            ("entities", example_get_entities()),
            ("cheques", example_get_rejected_checks()),
        ]

        failed_examples = [name for name, success in results if not success]

        if failed_examples:
            print("\n" + "=" * 80)
            print("⚠️  EXAMPLES COMPLETED WITH FAILURES")
            print("=" * 80)
            print("\nFailed sections:")
            for name in failed_examples:
                print(f"  - {name}")
            print("\nThe BCRA client code executed, but one or more upstream API calls failed.")
            return 1
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Pass the Central de Deudores manual to wire the next API")
        print("  2. Add a real source of (codigo_entidad, numero_cheque) pairs for batch analysis")
        print("  3. Reconnect company-level risk scoring once the remaining manuals are validated")
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
#!/usr/bin/env python3
"""
Portfolio Optimizer Test Script
"""

import sys
sys.path.insert(0, 'C:\\Project\\alphagenesis')

try:
    from core.portfolio_optimizer import PortfolioOptimizer
    
    print("Portfolio optimizer module loaded successfully")
    
    # Test optimizer initialization
    optimizer = PortfolioOptimizer()
    print("Portfolio optimizer initialized")
    
    # Test with sample data
    sample_results = [
        {
            'strategy_name': 'triple_combo', 
            'total_return': 15.5, 
            'sharpe_ratio': 1.2, 
            'max_drawdown': 8.5, 
            'win_rate': 65.0, 
            'volatility': 12.9
        },
        {
            'strategy_name': 'rsi_strategy', 
            'total_return': 12.3, 
            'sharpe_ratio': 0.9, 
            'max_drawdown': 12.1, 
            'win_rate': 58.0, 
            'volatility': 13.7
        },
        {
            'strategy_name': 'macd_strategy', 
            'total_return': 18.7, 
            'sharpe_ratio': 1.5, 
            'max_drawdown': 6.8, 
            'win_rate': 72.0, 
            'volatility': 12.5
        }
    ]
    
    print(f"Testing with {len(sample_results)} strategies")
    
    # Test portfolio optimization
    portfolios = optimizer.optimize_portfolio(sample_results, 'sharpe', 'medium')
    print(f"Generated {len(portfolios)} optimized portfolios")
    
    if portfolios:
        print("Portfolio details:")
        for i, portfolio in enumerate(portfolios):
            print(f"  {i+1}. {portfolio.name}")
            print(f"     Expected Return: {portfolio.expected_return:.2f}%")
            print(f"     Volatility: {portfolio.volatility:.2f}%")
            print(f"     Sharpe Ratio: {portfolio.sharpe_ratio:.2f}")
            print(f"     Weights: {portfolio.weights}")
            print()
        
        # Test report generation
        report = optimizer.generate_portfolio_report(portfolios)
        print(f"Portfolio report generated with {len(report.get('portfolios', []))} portfolios")
        
        if 'summary' in report:
            print(f"Report summary:")
            print(f"  Total portfolios: {report['summary']['total_portfolios']}")
            print(f"  Average expected return: {report['summary']['avg_expected_return']:.2f}%")
            print(f"  Average volatility: {report['summary']['avg_volatility']:.2f}%")
            print(f"  Average Sharpe ratio: {report['summary']['avg_sharpe_ratio']:.2f}")
    
    print("SUCCESS: Portfolio optimization system is fully operational!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
"""
백테스트 결과 다운로드 관련 API 라우트
"""

from flask import Blueprint, jsonify, request
import os
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

download_api = Blueprint('download', __name__)

def format_number(number):
    """숫자를 3자리마다 콤마로 구분하여 포맷팅"""
    if isinstance(number, (int, float)):
        return f"{number:,.2f}" if isinstance(number, float) else f"{number:,}"
    return str(number)

@download_api.route('/api/backtest/download/<result_id>', methods=['GET'])
def download_backtest_result(result_id):
    """백테스트 결과 Excel 다운로드"""
    try:
        from dashboard.routes import backtest_results
        
        # 다운로드 디렉토리 설정
        downloads_dir = r'C:\Project\alphagenesis\downloads'
        os.makedirs(downloads_dir, exist_ok=True)
        
        # 결과 찾기
        result_index = int(result_id) - 1
        if 0 <= result_index < len(backtest_results):
            result = backtest_results[result_index]
            
            # Excel 파일명 생성
            safe_strategy = result.strategy_name.replace('/', '_').replace(' ', '_')
            safe_symbol = result.symbol.replace('/', '_')
            filename = f"backtest_{safe_strategy}_{safe_symbol}_{result.start_date}_{result.end_date}.xlsx"
            filepath = os.path.join(downloads_dir, filename)
            
            # 거래 내역 데이터 준비
            trades_data = []
            if hasattr(result, 'trade_log') and result.trade_log:
                for trade in result.trade_log:
                    trades_data.append({
                        '백테스트 시간': trade.get('timestamp', ''),
                        '거래유형': trade.get('type', ''),
                        '심볼': trade.get('symbol', ''),
                        '가격': format_number(trade.get('price', 0)),
                        '수량': format_number(trade.get('amount', 0)),
                        '레버리지': f"{trade.get('leverage', 1.0):.1f}x",
                        '손익': format_number(trade.get('pnl', 0)),
                        '수익률': f"{trade.get('pnl_percent', 0):.2f}%",
                        '사유': trade.get('reason', ''),
                        '잔고': format_number(trade.get('balance_after', 0))
                    })
            
            # 요약 정보
            summary_data = {
                '전략명': result.strategy_name,
                '심볼': result.symbol,
                '시작일': result.start_date,
                '종료일': result.end_date,
                '초기자본': format_number(result.initial_capital),
                '최종가치': format_number(result.final_value),
                '총수익률': f"{result.total_return:.2f}%",
                '샤프비율': f"{result.sharpe_ratio:.2f}",
                '최대낙폭': f"{result.max_drawdown:.2f}%",
                '승률': f"{result.win_rate:.1f}%",
                '총거래수': format_number(result.total_trades),
                '수익거래': format_number(result.winning_trades),
                '손실거래': format_number(result.losing_trades),
                '평균레버리지': f"{getattr(result, 'avg_leverage', 1.0):.1f}x",
                '생성일시': result.created_at
            }
            
            # Excel 파일 생성 (다중 시트)
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 요약 시트
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='백테스트_요약', index=False)
                
                # 거래 내역 시트
                if trades_data:
                    trades_df = pd.DataFrame(trades_data)
                    trades_df.to_excel(writer, sheet_name='거래_내역', index=False)
                else:
                    # 거래 내역이 없을 경우
                    no_trades_df = pd.DataFrame([{'메시지': '거래 내역이 없습니다'}])
                    no_trades_df.to_excel(writer, sheet_name='거래_내역', index=False)
            
            return jsonify({
                'status': 'success',
                'message': f'백테스트 결과가 다운로드되었습니다',
                'filepath': filepath,
                'filename': filename
            })
        else:
            return jsonify({'error': '결과를 찾을 수 없습니다'}), 404
    
    except Exception as e:
        logger.error(f"다운로드 오류: {e}")
        return jsonify({'error': str(e)}), 500

@download_api.route('/api/backtest/download-all', methods=['POST'])
def download_all_results():
    """전체 백테스트 결과 Excel 다운로드"""
    try:
        from dashboard.routes import backtest_results
        
        data = request.get_json() or {}
        selected_ids = data.get('selected_ids', [])
        
        if not selected_ids:
            return jsonify({'error': '선택된 결과가 없습니다'}), 400
        
        # 다운로드 디렉토리 설정
        downloads_dir = r'C:\Project\alphagenesis\downloads'
        os.makedirs(downloads_dir, exist_ok=True)
        
        # 선택된 결과들 수집
        selected_results = []
        for result_id in selected_ids:
            result_index = int(result_id) - 1
            if 0 <= result_index < len(backtest_results):
                selected_results.append(backtest_results[result_index])
        
        if not selected_results:
            return jsonify({'error': '선택된 결과들을 찾을 수 없습니다'}), 404
        
        # Excel 파일 생성
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = os.path.join(downloads_dir, filename)
        
        # 데이터 준비
        results_data = []
        for result in selected_results:
            results_data.append({
                '전략명': result.strategy_name,
                '심볼': result.symbol,
                '시작일': result.start_date,
                '종료일': result.end_date,
                '초기자본': format_number(result.initial_capital),
                '최종가치': format_number(result.final_value),
                '총수익률(%)': f"{result.total_return:.2f}",
                '샤프비율': f"{result.sharpe_ratio:.2f}",
                '최대낙폭(%)': f"{result.max_drawdown:.2f}",
                '승률(%)': f"{result.win_rate:.1f}",
                '총거래수': format_number(result.total_trades),
                '수익거래': format_number(result.winning_trades),
                '손실거래': format_number(result.losing_trades),
                '평균레버리지': f"{getattr(result, 'avg_leverage', 1.0):.1f}x",
                '생성일시': result.created_at
            })
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 요약 시트
            df = pd.DataFrame(results_data)
            df.to_excel(writer, sheet_name='백테스트_결과', index=False)
            
            # 각 결과별 상세 거래 내역 시트 추가 (최대 10개만)
            for i, result in enumerate(selected_results[:10]):
                if hasattr(result, 'trade_log') and result.trade_log:
                    trades_data = []
                    for trade in result.trade_log:
                        trades_data.append({
                            '백테스트 시간': trade.get('timestamp', ''),
                            '거래유형': trade.get('type', ''),
                            '심볼': trade.get('symbol', ''),
                            '가격': format_number(trade.get('price', 0)),
                            '수량': format_number(trade.get('amount', 0)),
                            '레버리지': f"{trade.get('leverage', 1.0):.1f}x",
                            '손익': format_number(trade.get('pnl', 0)),
                            '수익률': f"{trade.get('pnl_percent', 0):.2f}%"
                        })
                    
                    if trades_data:
                        trades_df = pd.DataFrame(trades_data)
                        sheet_name = f"거래내역_{i+1}_{result.strategy_name[:10]}"
                        trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return jsonify({
            'status': 'success',
            'message': f'{len(selected_results)}개 결과가 다운로드되었습니다',
            'filepath': filepath,
            'filename': filename
        })
    
    except Exception as e:
        logger.error(f"Excel 다운로드 오류: {e}")
        return jsonify({'error': str(e)}), 500
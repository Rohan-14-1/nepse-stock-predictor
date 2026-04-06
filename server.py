#!/usr/bin/env python3
"""
NEPSE Intelligence Server
Serves the web app and provides analysis, company list, and market status API endpoints.
Dynamically fetches and updates all NEPSE-listed companies.
"""

import json
import sys
import os
import logging

from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from nepse_analysis import full_analysis, NumpyEncoder


class NEPSEHandler(SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logs

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self.serve_file('nepse_app.html', 'text/html')

        elif parsed.path == '/analyze' or parsed.path == '/api/analyze':
            params = parse_qs(parsed.query)
            ticker = params.get('ticker', ['NABIL'])[0].upper()

            try:
                logger.info(f"Starting analysis for {ticker}...")
                data = full_analysis(ticker)
                self.send_json(data)
                logger.info(f"Analysis complete for {ticker}")
            except Exception as e:
                logger.error(f"Analysis failed for {ticker}: {e}")
                self.send_json({'error': str(e)}, status=400)

        elif parsed.path == '/api/companies':
            try:
                from nepse_data_fetcher import fetch_company_list
                companies = fetch_company_list()
                
                # Group by sector
                sectors = {}
                for c in companies:
                    s = c.get('sector', 'Unknown')
                    if s not in sectors:
                        sectors[s] = []
                    sectors[s].append(c)

                self.send_json({
                    'companies': companies,
                    'sectors': {k: len(v) for k, v in sorted(sectors.items())},
                    'total': len(companies),
                    'source': 'live'
                })
            except ImportError:
                # Fallback: use the registry from engine
                from nepse_engine import get_company_registry
                registry = get_company_registry()
                companies = [
                    {'symbol': k, 'name': v.get('name', k), 'sector': v.get('sector', 'Unknown')}
                    for k, v in sorted(registry.items())
                ]
                self.send_json({
                    'companies': companies,
                    'sectors': {},
                    'total': len(companies),
                    'source': 'fallback'
                })
            except Exception as e:
                logger.error(f"Company list fetch failed: {e}")
                self.send_json({'error': str(e), 'companies': [], 'total': 0}, status=500)

        elif parsed.path == '/api/market-status':
            try:
                from nepse_data_fetcher import fetch_market_status
                status = fetch_market_status()
                self.send_json(status)
            except ImportError:
                self.send_json({
                    'is_open': False,
                    'source': 'offline',
                    'message': 'Data fetcher not available'
                })
            except Exception as e:
                self.send_json({'error': str(e)}, status=500)

        elif parsed.path == '/api/refresh-companies':
            try:
                from nepse_data_fetcher import fetch_company_list
                companies = fetch_company_list(force_refresh=True)
                self.send_json({
                    'companies': companies,
                    'total': len(companies),
                    'refreshed': True
                })
            except Exception as e:
                self.send_json({'error': str(e)}, status=500)

        elif parsed.path.endswith('.html'):
            filename = parsed.path.lstrip('/')
            self.serve_file(filename, 'text/html')

        elif parsed.path.endswith('.css'):
            filename = parsed.path.lstrip('/')
            self.serve_file(filename, 'text/css')

        elif parsed.path.endswith('.js'):
            filename = parsed.path.lstrip('/')
            self.serve_file(filename, 'application/javascript')

        else:
            self.send_response(404)
            self.end_headers()

    def serve_file(self, filename, content_type):
        try:
            with open(filename, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data, status=200):
        content = json.dumps(data, cls=NumpyEncoder).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)


def run_server(port=8765):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(('0.0.0.0', port), NEPSEHandler)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       NEPSE Intelligence — Stock Analysis Server        ║
║                                                         ║
║   🌐  http://localhost:{port}                            ║
║   📡  Live data from Nepal Stock Exchange               ║
║   🤖  ML-powered profit/loss prediction                 ║
║                                                         ║
║   API Endpoints:                                        ║
║     GET /api/companies      — All listed companies      ║
║     GET /api/analyze?ticker=NABIL — Run analysis        ║
║     GET /api/market-status  — Market status             ║
╚══════════════════════════════════════════════════════════╝
    """)
    server.serve_forever()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(port)

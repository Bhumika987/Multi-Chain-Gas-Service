# Multi-Chain Gas Tracker API 🌐⛽

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Web3.py](https://img.shields.io/badge/Web3.py-7.x-brightgreen.svg)](https://web3py.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-blue)](https://fastapi.tiangolo.com/)

A comprehensive API for real-time gas tracking across multiple EVM-compatible chains with EIP-1559 support, historical data, and transaction simulation.

![API Dashboard Screenshot](https://via.placeholder.com/800x400?text=API+Dashboard+Screenshot)

## ✨ Key Features

- **Multi-Chain Support**: Ethereum, Base, zkSync Era, Avalanche, Fantom, Polygon zkEVM, Moonbeam
- **EIP-1559 Support**: Base fee and priority fee calculations
- **Historical Data**: Track base fee trends over time
- **L2 Fee Breakdown**: Detailed L1/L2 fee analysis for Layer 2 chains
- **Transaction Simulation**: Dry-run transactions before sending
- **Rate Limited API**: Built-in request throttling
- **Web Dashboard**: Interactive UI for testing endpoints

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- SQLite (for historical data storage)

```bash
# Clone repository
git clone https://github.com/Bhumika987/Multi-Chain-Gas-Tracker.git
cd Multi-Chain-Gas-Tracker

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Infura/API keys
```

## 🚀 Usage

### Running the API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

### Accessing the Web UI
The dashboard will be available at `http://localhost:8000/ui`

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service health check |
| `/chains` | GET | List all supported chains |
| `/chain-info` | GET | Get chain metadata |
| `/gas` | GET | Current gas prices |
| `/estimate-fee` | GET | Transaction fee estimation |
| `/estimate-fee/l2-breakdown` | GET | L2 fee breakdown |
| `/historical/base-fee` | GET | Historical base fee data |
| `/balance/{address}` | GET | Native token balance |
| `/simulate` | POST | Transaction simulation |

## 🌐 Supported Chains

| Chain | Chain ID | Native Token | EIP-1559 |
|-------|----------|--------------|----------|
| Ethereum | 1 | ETH | ✅ |
| Base | 8453 | ETH | ✅ |
| zkSync Era | 324 | ETH | ✅|
| Avalanche C-Chain | 43114 | AVAX | ✅ |
| Fantom | 250 | FTM | ✅ |
| Polygon zkEVM | 1101 | ETH | ✅ |
| Moonbeam | 1284 | GLMR | ✅ |

## 📂 Project Structure

```
multi-chain-gas-tracker/
├── src/
│   ├── main.py                # FastAPI application
│   ├── poa_middleware.py      # Proof-of-Authority middleware
│   ├── chains.json            # Chain configurations
│   └── historical_gas.db      # SQLite database for historical data
├── static/                    # Web UI assets
│   └── index.html             # Dashboard HTML
├── .env                       # Environment variables
├── .env.example               # Example env file
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📬 Contact

Bhumika - [@Bhumika987](https://github.com/Bhumika987) - bhumika@example.com

🔗 **Project Link**: [https://github.com/Bhumika987/Multi-Chain-Gas-Tracker](https://github.com/Bhumika987/Multi-Chain-Gas-Tracker)

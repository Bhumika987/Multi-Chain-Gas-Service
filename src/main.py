from fastapi.staticfiles import StaticFiles  
from fastapi.responses import FileResponse 
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict, Union, List
from dotenv import load_dotenv
from web3 import Web3
from .poa_middleware import geth_poa_middleware
from web3.exceptions import ContractLogicError
from decimal import Decimal, getcontext
from cachetools import TTLCache
import sqlite3
import pandas as pd
from contextlib import asynccontextmanager


# Configure decimal precision
getcontext().prec = 18


# Load environment variables
load_dotenv()


# Cache configuration
ETH_PRICE_CACHE_TTL = 60
GAS_PRICE_CACHE_TTL = 15
EIP1559_CACHE_TTL = 12
BALANCE_CACHE_TTL = 30
HISTORICAL_CACHE_TTL = 300  # 5 minutes for historical data


# Initialize caches
eth_price_cache = TTLCache(maxsize=1, ttl=ETH_PRICE_CACHE_TTL)
gas_price_cache = TTLCache(maxsize=100, ttl=GAS_PRICE_CACHE_TTL)
eip1559_cache = TTLCache(maxsize=100, ttl=EIP1559_CACHE_TTL)
balance_cache = TTLCache(maxsize=1000, ttl=BALANCE_CACHE_TTL)
historical_cache = TTLCache(maxsize=50, ttl=HISTORICAL_CACHE_TTL)


# Database setup for historical data
HISTORICAL_DB = str(Path(__file__).parent / "../historical_gas.db")


def init_db():
    conn = sqlite3.connect(HISTORICAL_DB)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS base_fee_history (
        chain_id INTEGER,
        block_number INTEGER,
        base_fee_gwei REAL,
        timestamp INTEGER,
        PRIMARY KEY (chain_id, block_number)
    )
    """)
    conn.commit()
    conn.close()


init_db()


# Load chain config
try:
    config_path = Path(__file__).parent / "chains.json"
    with open(config_path, "r") as f:
        CHAINS = json.load(f)["chains"]
        CHAIN_MAP = {chain["chainId"]: chain for chain in CHAINS}
except FileNotFoundError:
    raise RuntimeError("Missing chains.json config file")
except json.JSONDecodeError:
    raise RuntimeError("Invalid JSON format in chains.json")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup code
    import web3 as web3_pkg
    print(f"Using web3.py version: {web3_pkg.__version__}")
    
    for chain in CHAINS:
        try:
            w3 = get_web3_connection(chain["chainId"])
            if w3.is_connected():
                supports_eip1559 = 'baseFeePerGas' in w3.eth.get_block('latest')
                print(f"✅ Connected to {chain['chainName']} (ID: {chain['chainId']})")
                print(f"    EIP-1559 Support: {'Yes' if supports_eip1559 else 'No'}")
                
                if supports_eip1559:
                    try:
                        block = w3.eth.get_block('latest')
                        store_base_fee(chain['chainId'], block['number'], block['baseFeePerGas'])
                        print(f"    Initialized historical tracking")
                    except Exception as e:
                        print(f"    ⚠️  Failed to initialize historical tracking: {str(e)}")
        except Exception as e:
            print(f"❌ Failed to connect to {chain['chainName']}: {str(e)}")
    yield
    # Shutdown code would go here (optional)

#initialize FASTAPI with rate limiting
app = FastAPI(
    title="Multi-Chain Gas Tracker API",
    version="5.0",
    description="API for gas tracking with EIP-1559, historical data, and L2 support",
    docs_url="/docs",
    lifespan=lifespan
)


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    raise HTTPException(status_code=429, detail="Too many requests")




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui", include_in_schema=False)
async def serve_ui():
    return FileResponse("static/index.html")

# Data Models
class GasPriceResponse(BaseModel):
    legacy: Optional[float] = Field(None, description="Legacy gas price in Gwei")
    baseFee: Optional[float] = Field(None, description="Current base fee in Gwei")
    maxPriorityFee: Optional[float] = Field(None, description="Suggested max priority fee in Gwei")
    maxFee: Optional[float] = Field(None, description="Suggested max fee in Gwei")
    source: str = Field(..., description="Data source provider")
    timestamp: float = Field(..., description="Time of data retrieval")


class FeeEstimationResponse(BaseModel):
    gas_limit: int = Field(..., description="Estimated gas units needed")
    gas_price: Optional[int] = Field(None, description="Gas price in wei (legacy)")
    max_fee: Optional[int] = Field(None, description="Max fee per gas (EIP-1559)")
    max_priority_fee: Optional[int] = Field(None, description="Max priority fee per gas (EIP-1559)")
    total_fee: int = Field(..., description="Total fee in wei")
    total_fee_eth: float = Field(..., description="Total fee in ETH")
    total_fee_usd: Optional[float] = Field(None, description="Estimated fee in USD")
    warning: Optional[str] = Field(None, description="Any important notices")


class HistoricalFeeData(BaseModel):
    timestamp: int
    base_fee_gwei: float
    block_number: int


class HistoricalFeeResponse(BaseModel):
    chain_id: int
    data: List[HistoricalFeeData]
    time_range_hours: int


class L2FeeResponse(BaseModel):
    l1_fee: Optional[float] = Field(None, description="Estimated L1 security fee")
    l2_fee: Optional[float] = Field(None, description="L2 execution fee")
    total_fee: float
    l1_fee_percentage: Optional[float] = Field(None, description="Percentage of total fee from L1")


class BalanceResponse(BaseModel):
    address: str = Field(..., description="Ethereum address")
    balance_wei: int = Field(..., description="Balance in wei")
    balance_eth: float = Field(..., description="Balance in ETH")
    balance_usd: Optional[float] = Field(None, description="Estimated balance in USD")


class SimulateRequest(BaseModel):
    from_address: str = Field(..., description="Sender address (0x...)")
    to_address: str = Field(..., description="Contract/recipient address")
    value: int = Field(0, description="Amount in wei")
    data: Optional[str] = Field(None, description="Call data (hex encoded)")
    gas_limit: Optional[int] = Field(None, description="Optional gas limit")
    chain_id: Optional[int] = Field(1, description="Chain ID for the transaction")
    max_fee: Optional[int] = Field(None, description="Max fee per gas (EIP-1559)")
    max_priority_fee: Optional[int] = Field(None, description="Max priority fee per gas (EIP-1559)")


class SimulateResponse(BaseModel):
    success: bool = Field(..., description="True if the tx would succeed")
    gas_used: int = Field(..., description="Actual gas consumed")
    error: Optional[str] = Field(None, description="Revert reason if failed")
    traces: Optional[Dict] = Field(None, description="Debug traces (optional)")


class ChainInfoResponse(BaseModel):
    chainName: str
    chainId: int
    currentGasPrice: str
    nativeToken: str
    supportsEIP1559: bool = Field(..., description="Whether chain supports EIP-1559")


class SupportedChainsResponse(BaseModel):
    chains: List[Dict]


# Helper Classes
class EnhancedGasEstimator:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self.default_gas_limits = {
            'simple_transfer': 21000,
            'token_transfer': 65000,
            'contract_interaction': 100000,
            'complex_contract': 200000
        }


    def estimate_gas_limit(self, params: Dict) -> int:
        """Smart gas estimation with fallback logic"""
        try:
            return self.web3.eth.estimate_gas(params)
        except ContractLogicError as e:
            print(f"Contract logic error: {str(e)}")
            return self._get_default_gas_limit(params.get('data'))
        except Exception as e:
            print(f"Estimation failed: {str(e)}")
            return self._get_default_gas_limit(params.get('data'))


    def _get_default_gas_limit(self, data: Optional[str]) -> int:
        """Get appropriate default based on transaction type"""
        if not data or data == '0x':
            return self.default_gas_limits['simple_transfer']
        return self.default_gas_limits['contract_interaction']


# Helper Functions
def get_web3_connection(chain_id: int = 1) -> Web3:
    """Initialize Web3 for a specific chain with retry mechanism"""
    chain = CHAIN_MAP.get(chain_id)
    if not chain:
        raise ValueError(f"Chain ID {chain_id} not supported")
   
    rpc_url = chain["rpcUrl"]
    if not rpc_url:
        raise ValueError(f"No RPC URL configured for chain {chain_id}")
   
    for attempt in range(3):
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url))
           
            if w3.is_connected():
                return w3
            time.sleep(1)
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
    raise ConnectionError(f"Could not connect to chain {chain_id} at {rpc_url}")
       
def get_eth_usd_price() -> Optional[float]:
    """Fetch current ETH price in USD with caching"""
    try:
        if 'eth_price' in eth_price_cache:
            return eth_price_cache['eth_price']
           
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd",
            timeout=3
        )
        response.raise_for_status()
        price = response.json().get("ethereum", {}).get("usd")
       
        if price is not None:
            eth_price_cache['eth_price'] = price
        return price
    except Exception as e:
        print(f"Failed to fetch ETH price: {str(e)}")
        return None


def get_legacy_gas(chain_id: int = 1) -> GasPriceResponse:
    """Fetch legacy gas price from RPC endpoint"""
    try:
        w3 = get_web3_connection(chain_id)
        gas_price = w3.eth.gas_price
        gwei = w3.from_wei(gas_price, 'gwei')
        return GasPriceResponse(
            legacy=float(gwei),
            baseFee=None,
            maxPriorityFee=None,
            maxFee=None,
            source="rpc",
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RPC error: {str(e)}")


def get_historical_base_fees(chain_id: int, hours: int = 24) -> List[HistoricalFeeData]:
    cache_key = f"historical_{chain_id}_{hours}"
    if cache_key in historical_cache:
        return historical_cache[cache_key]
   
    conn = sqlite3.connect(HISTORICAL_DB)
    try:
        cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())
        query = """
            SELECT block_number, base_fee_gwei, timestamp
            FROM base_fee_history
            WHERE chain_id = ? AND timestamp >= ?
            ORDER BY block_number DESC
        """
        df = pd.read_sql(query, conn, params=(chain_id, cutoff))
        data = [
            HistoricalFeeData(
                timestamp=row['timestamp'],
                base_fee_gwei=row['base_fee_gwei'],
                block_number=row['block_number']
            )
            for _, row in df.iterrows()
        ]
        historical_cache[cache_key] = data
        return data
    finally:
        conn.close()


def store_base_fee(chain_id: int, block_number: int, base_fee: int):
    conn = sqlite3.connect(HISTORICAL_DB)
    try:
        base_fee_gwei = Web3.from_wei(base_fee, 'gwei')
        timestamp = int(time.time())
        conn.execute(
            "INSERT OR REPLACE INTO base_fee_history VALUES (?, ?, ?, ?)",
            (chain_id, block_number, float(base_fee_gwei), timestamp)
        )
        conn.commit()
    finally:
        conn.close()


def calculate_optimism_fees(w3: Web3, tx_params: Dict) -> Dict:
    """Optimism specific fee calculation"""
    l1_fee = w3.eth.estimate_gas({**tx_params, 'gasPrice': 0})
    l2_fee = w3.eth.estimate_gas(tx_params) - l1_fee
    return {
        'l1_fee': l1_fee,
        'l2_fee': l2_fee,
        'total_fee': l1_fee + l2_fee,
        'l1_fee_percentage': (l1_fee / (l1_fee + l2_fee)) * 100
    }


def calculate_arbitrum_fees(w3: Web3, tx_params: Dict) -> Dict:
    """Arbitrum uses a different gas model"""
    total_gas = w3.eth.estimate_gas(tx_params)
    l1_fee = total_gas * 0.3  # Simplified approximation
    return {
        'l1_fee': l1_fee,
        'l2_fee': total_gas - l1_fee,
        'total_fee': total_gas,
        'l1_fee_percentage': 30  # Rough estimate
    }


def get_cached_gas_price(chain_id: int) -> GasPriceResponse:
    """Get gas price with caching"""
    cache_key = f"gas_price_{chain_id}"
    if cache_key in gas_price_cache:
        return gas_price_cache[cache_key]
   
    gas_data = get_legacy_gas(chain_id)
    gas_price_cache[cache_key] = gas_data
    return gas_data


def get_cached_eip1559_gas(chain_id: int) -> GasPriceResponse:
    """Get EIP-1559 gas data with caching"""
    cache_key = f"eip1559_{chain_id}"
    if cache_key in eip1559_cache:
        return eip1559_cache[cache_key]
   
    gas_data = get_eip1559_gas(chain_id)
    eip1559_cache[cache_key] = gas_data
    return gas_data


def get_eip1559_gas(chain_id: int = 1) -> GasPriceResponse:
    try:
        w3 = get_web3_connection(chain_id)
        current_block = w3.eth.get_block('latest')
       
        if not CHAIN_MAP[chain_id].get('supportsEIP1559', False):
            return get_legacy_gas(chain_id)
       
        base_fee = current_block['baseFeePerGas']
        store_base_fee(chain_id, current_block['number'], base_fee)
       
        # Chain-specific priority fee logic
        if chain_id == 137:  # Polygon
            priority_fee = w3.to_wei(30, 'gwei')  # Polygon typically needs higher tips
        else:
            priority_fee = w3.eth.max_priority_fee or w3.to_wei(2, 'gwei')
       
        base_fee_gwei = w3.from_wei(base_fee, 'gwei')
        priority_fee_gwei = w3.from_wei(priority_fee, 'gwei')
        max_fee_gwei = float(base_fee_gwei * 2 + priority_fee_gwei)
       
        return GasPriceResponse(
            legacy=None,
            baseFee=float(base_fee_gwei),
            maxPriorityFee=float(priority_fee_gwei),
            maxFee=float(max_fee_gwei),
            source="rpc",
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RPC error: {str(e)}")


def get_cached_balance(address: str, chain_id: int) -> int:
    """Get balance with caching"""
    cache_key = f"balance_{address.lower()}_{chain_id}"
    if cache_key in balance_cache:
        return balance_cache[cache_key]
   
    w3 = get_web3_connection(chain_id)
    checksum_addr = w3.to_checksum_address(address)
    balance = w3.eth.get_balance(checksum_addr)
    balance_cache[cache_key] = balance
    return balance


# API Endpoints
@app.get("/", summary="Service Status", tags=["Health Check"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Service health check endpoint"""
    return {
        "status": "online",
        "version": app.version,
        "services": {
            "supported_chains": [c["chainName"] for c in CHAINS],
            "etherscan": bool(os.getenv("ETHERSCAN_API_KEY")),
            "eip1559_support": True,
            "rate_limited": True
        }
    }

@app.get("/health")
def render_health_check():
    """Simplified health check for Render monitoring"""
    return {"status": "ok"}


@app.get("/chains", response_model=SupportedChainsResponse, summary="List supported chains", tags=["Chain Info"])
async def get_supported_chains():
    """Get list of all supported chains"""
    return {"chains": CHAINS}


@app.get("/chain-info", response_model=ChainInfoResponse, summary="Get chain info", tags=["Chain Info"])
async def get_chain_info(
    chain_name: str = Query(None, description="Chain name"),
    chain_id: int = Query(None, description="Chain ID")
):
    """Get gas price and native token info for a chain"""
    chain = None
    if chain_name:
        chain = next((c for c in CHAINS if c["chainName"].lower() == chain_name.lower()), None)
    elif chain_id:
        chain = CHAIN_MAP.get(chain_id)
   
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")


    try:
        w3 = get_web3_connection(chain["chainId"])
        current_block = w3.eth.get_block('latest')
        supports_eip1559 = 'baseFeePerGas' in current_block
        gas_price = current_block['baseFeePerGas'] if supports_eip1559 else w3.eth.gas_price
       
        return {
            "chainName": chain["chainName"],
            "chainId": chain["chainId"],
            "currentGasPrice": str(gas_price),
            "nativeToken": chain["nativeToken"],
            "supportsEIP1559": supports_eip1559
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chain info: {str(e)}")


@app.get("/gas", response_model=GasPriceResponse, summary="Get current gas prices", tags=["Gas Data"])
@limiter.limit("30/minute")
async def get_gas_prices(
        request: Request,
        source: str = Query("rpc", description="Data source: 'rpc'"),
        chain_id: int = Query(1, description="Chain ID"),
        eip1559: bool = Query(True, description="Include EIP-1559 data when available")
):
    """Fetch current network gas prices"""
    if source == "rpc":
        try:
            if eip1559:
                return get_cached_eip1559_gas(chain_id)
        except:
            pass
        return get_cached_gas_price(chain_id)
    else:
        raise HTTPException(status_code=400, detail="Unsupported source")
   
@app.get("/estimate-fee", response_model=FeeEstimationResponse, summary="Estimate transaction fees", tags=["Transaction"])
@limiter.limit("20/minute")
async def estimate_fee(request: Request,
    from_address: str = Query(..., description="Sender address (0x...)"),
    to_address: str = Query(..., description="Recipient address (0x...)"),
    value: int = Query(0, description="Amount to send in wei"),
    data: Optional[str] = Query(None, description="Transaction data payload"),
    gas_price: Optional[int] = Query(None, description="Custom gas price in wei (legacy)"),
    max_fee: Optional[int] = Query(None, description="Max fee per gas (EIP-1559)"),
    max_priority_fee: Optional[int] = Query(None, description="Max priority fee per gas (EIP-1559)"),
    chain_id: int = Query(1, description="Chain ID")
):
    """Estimate transaction fees for a specific chain"""
    try:
        w3 = get_web3_connection(chain_id)
        estimator = EnhancedGasEstimator(w3)
        eth_price = get_eth_usd_price()
       
        from_address = w3.to_checksum_address(from_address)
        to_address = w3.to_checksum_address(to_address)
       
        params = {
            'from': from_address,
            'to': to_address,
            'value': value,
            'data': data or '0x'
        }
       
        current_block = w3.eth.get_block('latest')
        is_eip1559 = 'baseFeePerGas' in current_block
       
        if is_eip1559 and (max_fee or max_priority_fee or not gas_price):
            if not max_priority_fee:
                try:
                    max_priority_fee = w3.eth.max_priority_fee
                except:
                    max_priority_fee = w3.to_wei(2, 'gwei')
                   
            if not max_fee:
                base_fee = current_block['baseFeePerGas']
                max_fee = base_fee * 2 + max_priority_fee
               
            params['maxFeePerGas'] = max_fee
            params['maxPriorityFeePerGas'] = max_priority_fee
            gas_price = None
        else:
            if not gas_price:
                gas_price = w3.eth.gas_price
            params['gasPrice'] = gas_price
       
        gas_limit = estimator.estimate_gas_limit(params)
        total_fee = gas_limit * (max_fee if is_eip1559 else gas_price)
        total_fee_eth = float(w3.from_wei(total_fee, 'ether'))
       
        return FeeEstimationResponse(
            gas_limit=gas_limit,
            gas_price=gas_price,
            max_fee=max_fee,
            max_priority_fee=max_priority_fee,
            total_fee=total_fee,
            total_fee_eth=total_fee_eth,
            total_fee_usd=total_fee_eth * eth_price if eth_price else None,
            warning="Using EIP-1559 fees" if is_eip1559 else "Using legacy gas price"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fee estimation failed: {str(e)}")
   
@app.get("/estimate-fee/l2-breakdown", response_model=L2FeeResponse, tags=["L2 Support"])
async def estimate_l2_fee_breakdown(
    from_address: str = Query(..., description="Sender address"),
    to_address: str = Query(..., description="Recipient address"),
    chain_id: int = Query(..., description="L2 Chain ID"),
    value: int = Query(0, description="Amount in wei"),
    data: Optional[str] = Query(None, description="Transaction data")
):
    """Get detailed L2 fee breakdown (L1 security fee + L2 execution fee)"""
    chain = CHAIN_MAP.get(chain_id)
    if not chain or not chain.get('isL2', False):
        raise HTTPException(status_code=400, detail="Not an L2 chain")
   
    w3 = get_web3_connection(chain_id)
    params = {
        'from': w3.to_checksum_address(from_address),
        'to': w3.to_checksum_address(to_address),
        'value': value,
        'data': data or '0x'
    }
   
    try:
        if chain_id == 10:  # Optimism
            result = calculate_optimism_fees(w3, params)
        elif chain_id == 42161:  # Arbitrum
            result = calculate_arbitrum_fees(w3, params)
        else:
            result = {'total_fee': w3.eth.estimate_gas(params)}
       
        return L2FeeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")


@app.get("/historical/base-fee", response_model=HistoricalFeeResponse, tags=["Historical Data"])
async def get_historical_base_fee(
    chain_id: int = Query(..., description="Chain ID"),
    hours: int = Query(24, description="Time window in hours (max 168)")
):
    """Get historical base fee data for EIP-1559 chains"""
    if chain_id not in CHAIN_MAP:
        raise HTTPException(status_code=404, detail="Chain not found")
    if not CHAIN_MAP[chain_id].get('supportsEIP1559', False):
        raise HTTPException(status_code=400, detail="Chain does not support EIP-1559")
    if hours > 168:
        hours = 168
   
    data = get_historical_base_fees(chain_id, hours)
    return HistoricalFeeResponse(
        chain_id=chain_id,
        data=data,
        time_range_hours=hours
    )


@app.get("/balance/{address}", response_model=BalanceResponse, summary="Get native token balance", tags=["Wallet"])
async def get_balance(
    address: str,
    chain_id: int = Query(1, description="Chain ID")
):
    """Get native token balance for an address on a specific chain"""
    try:
        w3 = get_web3_connection(chain_id)
        balance_wei = get_cached_balance(address, chain_id)
        balance_eth = balance_wei / 10**18
        eth_price_USD = get_eth_usd_price()
       
        return BalanceResponse(
            address=w3.to_checksum_address(address),
            balance_wei=balance_wei,
            balance_eth=balance_eth,
            balance_usd=balance_eth * eth_price_USD if eth_price_USD else None
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Balance check failed: {str(e)}")


@app.post("/simulate", response_model=SimulateResponse, summary="Dry-run a transaction", tags=["Transaction Simulation"])
@limiter.limit("10/minute")
async def simulate_transaction(request: SimulateRequest):
    """Simulate a transaction on a specific chain"""
    try:
        w3 = get_web3_connection(request.chain_id)
        from_addr = w3.to_checksum_address(request.from_address)
        to_addr = w3.to_checksum_address(request.to_address) if request.to_address else None
       
        tx_params = {
            "from": from_addr,
            "to": to_addr,
            "value": request.value,
            "data": request.data or "0x",
        }
       
        current_block = w3.eth.get_block('latest')
        is_eip1559 = 'baseFeePerGas' in current_block
       
        if is_eip1559 and (request.max_fee or request.max_priority_fee):
            if not request.max_priority_fee:
                request.max_priority_fee = w3.eth.max_priority_fee
            if not request.max_fee:
                base_fee = current_block['baseFeePerGas']
                request.max_fee = base_fee * 2 + request.max_priority_fee
               
            tx_params['maxFeePerGas'] = request.max_fee
            tx_params['maxPriorityFeePerGas'] = request.max_priority_fee
        else:
            tx_params['gasPrice'] = w3.eth.gas_price
       
        gas_limit = request.gas_limit or w3.eth.estimate_gas(tx_params)
       
        try:
            result = w3.eth.call(tx_params, block_identifier="latest")
            return SimulateResponse(
                success=True,
                gas_used=gas_limit,
            )
        except ContractLogicError as e:
            return SimulateResponse(
                success=False,
                gas_used=gas_limit,
                error=str(e),
            )
           
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
   
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="src")

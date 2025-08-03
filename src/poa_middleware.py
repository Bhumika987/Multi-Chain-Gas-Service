from web3.types import RPCEndpoint, RPCResponse

def geth_poa_middleware(make_request, w3):
    """
    Middleware for handling Proof-of-Authority chains
    """
    def middleware(method: RPCEndpoint, params):
        # Process the request normally first
        response = make_request(method, params)
        
        # Handle POA-specific block responses
        if method in ('eth_getBlockByHash', 'eth_getBlockByNumber'):
            if response and 'result' in response and response['result']:
                block = response['result']
                if 'extraData' in block and len(block['extraData']) > 66:
                    # Trim extraData to 32 bytes for POA compatibility
                    block['extraData'] = block['extraData'][:66]
        
        return response
    
    return middleware
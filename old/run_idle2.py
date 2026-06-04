import asyncio, sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'step3_vtube'))
import pyvts

async def main():
    plugin_info = {
        'plugin_name': 'emeth-controller',
        'developer': 'emeth',
        'authentication_token_path': './vts_token2.json'
    }
    vts = pyvts.vts(plugin_info=plugin_info)
    await vts.connect()
    
    # Request token
    await vts.request(vts.vts_request.requestToken())
    print('Waiting for Allow in VTube Studio...')
    
    # Wait and read token
    response = await vts.request(vts.vts_request.requestToken())
    token = response['data']['authenticationToken']
    with open('vts_token2.json', 'w') as f:
        json.dump({'token': token}, f)
    
    await vts.request(vts.vts_request.requestAuthentication(token=token))
    print('Authenticated!')
    
    # Send idle animation params for 15 seconds
    t0 = time.time()
    print('Starting idle animation...')
    while time.time() - t0 < 15:
        t = time.time() - t0
        breath = 0.5 + 0.5 * math.sin(2 * math.pi * t / 4)
        head_x = 10 * math.sin(2 * math.pi * t / 6)
        head_y = 5 * math.sin(2 * math.pi * t / 7)
        blink = 0 if (int(t * 2) % 8 == 0) else 1
        
        params = [
            {'id': 'ParamBreath', 'value': breath},
            {'id': 'ParamAngleX', 'value': head_x},
            {'id': 'ParamAngleY', 'value': head_y},
            {'id': 'ParamEyeLOpen', 'value': blink},
            {'id': 'ParamEyeROpen', 'value': blink},
        ]
        
        request = vts.vts_request.BaseRequest('InjectParameterDataRequest', {
            'parameterValues': [{'id': p['id'], 'value': p['value']} for p in params]
        })
        await vts.request(request)
        await asyncio.sleep(0.033)
    
    print('Done!')
    await vts.close()

asyncio.run(main())

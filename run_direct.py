import asyncio, json, math, time, websockets

async def main():
    uri = 'ws://localhost:8001'
    async with websockets.connect(uri) as ws:
        # Request token
        await ws.send(json.dumps({
            'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
            'requestID': 'r1', 'messageType': 'AuthenticationTokenRequest',
            'data': {'pluginName': 'idle-demo', 'pluginDeveloper': 'emeth'}
        }))
        resp = json.loads(await ws.recv())
        token = resp['data'].get('authenticationToken', '')
        if not token:
            print('Need to Allow in VTube Studio!')
            return
        
        # Authenticate
        await ws.send(json.dumps({
            'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
            'requestID': 'r2', 'messageType': 'AuthenticationRequest',
            'data': {'pluginName': 'idle-demo', 'pluginDeveloper': 'emeth', 'authenticationToken': token}
        }))
        resp = json.loads(await ws.recv())
        if resp['data'].get('authenticated'):
            print('Authenticated!')
        else:
            print('Auth failed:', resp['data'].get('reason'))
            return
        
        # Inject params for 20 seconds
        t0 = time.time()
        print('Injecting idle animation...')
        while time.time() - t0 < 20:
            t = time.time() - t0
            breath = 0.5 + 0.5 * math.sin(2 * math.pi * t / 4)
            head_x = 15 * math.sin(2 * math.pi * t / 5)
            head_y = 8 * math.sin(2 * math.pi * t / 7)
            head_z = 5 * math.sin(2 * math.pi * t / 9)
            blink = 0.0 if int(t * 10) % 40 < 3 else 1.0
            
            await ws.send(json.dumps({
                'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
                'requestID': 'r3', 'messageType': 'InjectParameterDataRequest',
                'data': {'parameterValues': [
                    {'id': 'ParamBreath', 'value': breath},
                    {'id': 'ParamAngleX', 'value': head_x},
                    {'id': 'ParamAngleY', 'value': head_y},
                    {'id': 'ParamAngleZ', 'value': head_z},
                    {'id': 'ParamEyeLOpen', 'value': blink},
                    {'id': 'ParamEyeROpen', 'value': blink},
                ]}
            }))
            await ws.recv()
            await asyncio.sleep(0.033)
        print('Done!')

asyncio.run(main())

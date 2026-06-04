import asyncio, json, math, time, websockets

async def main():
    uri = 'ws://localhost:8001'
    async with websockets.connect(uri) as ws:
        # Request token
        await ws.send(json.dumps({
            'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
            'requestID': 'r1', 'messageType': 'AuthenticationTokenRequest',
            'data': {'pluginName': 'arm-fixer', 'pluginDeveloper': 'emeth'}
        }))
        resp = json.loads(await ws.recv())
        token = resp['data'].get('authenticationToken', '')
        if not token:
            print('Waiting for Allow popup in VTube Studio...')
            # Wait up to 30 seconds for popup
            await asyncio.sleep(30)
            return

        # Authenticate
        await ws.send(json.dumps({
            'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
            'requestID': 'r2', 'messageType': 'AuthenticationRequest',
            'data': {'pluginName': 'arm-fixer', 'pluginDeveloper': 'emeth', 'authenticationToken': token}
        }))
        resp = json.loads(await ws.recv())
        if not resp['data'].get('authenticated'):
            print('Auth failed:', resp['data'].get('reason'))
            return
        print('Authenticated!')

        # Get ArtMesh list
        await ws.send(json.dumps({
            'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
            'requestID': 'r3', 'messageType': 'ArtMeshListRequest',
            'data': {}
        }))
        resp = json.loads(await ws.recv())
        meshes = resp['data'].get('artMeshNames', [])
        tags = resp['data'].get('artMeshTags', [])
        
        arm_b_meshes = []
        for i, name in enumerate(meshes):
            if 'ArmRB' in name or 'ArmLB' in name or 'armRB' in name or 'armLB' in name:
                arm_b_meshes.append(name)
                print(f'B-arm mesh: {name}')
        
        if not arm_b_meshes:
            print('No B-arm meshes found. Checking all arm meshes:')
            for name in meshes:
                if 'arm' in name.lower() or 'Arm' in name:
                    print(f'  {name}')

        # Hide B-arm artmeshes by setting opacity to 0
        if arm_b_meshes:
            tint_data = []
            for mesh in arm_b_meshes:
                tint_data.append({
                    'jeb_': False,
                    'artMeshMatcher': {'tintAll': False, 'artMeshNumber': [meshes.index(mesh)]},
                    'colorTint': {'colorR': 0, 'colorG': 0, 'colorB': 0, 'colorA': 0, 'mixWithSceneLightingColor': 0.0}
                })
            await ws.send(json.dumps({
                'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
                'requestID': 'r4', 'messageType': 'ColorTintRequest',
                'data': {'colorTint': {'colorR': 255, 'colorG': 255, 'colorB': 255, 'colorA': 0}, 'artMeshMatcher': {'tintAll': False, 'nameExact': arm_b_meshes}}
            }))
            resp = json.loads(await ws.recv())
            print(f'Tint result: {resp["data"]}')

        # Also run idle animation for 20 seconds
        t0 = time.time()
        print('Starting idle with fixed arms for 20 seconds...')
        while time.time() - t0 < 20:
            t = time.time() - t0
            breath = 0.5 + 0.5 * math.sin(2 * math.pi * t / 4)
            head_x = 15 * math.sin(2 * math.pi * t / 5)
            head_y = 8 * math.sin(2 * math.pi * t / 7)
            head_z = 5 * math.sin(2 * math.pi * t / 9)
            blink = 0.0 if int(t * 10) % 40 < 3 else 1.0

            # Re-hide B arms each frame (tint resets)
            await ws.send(json.dumps({
                'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
                'requestID': 'rt', 'messageType': 'ColorTintRequest',
                'data': {'colorTint': {'colorR': 255, 'colorG': 255, 'colorB': 255, 'colorA': 0}, 'artMeshMatcher': {'tintAll': False, 'nameExact': arm_b_meshes}}
            }))
            await ws.recv()

            await ws.send(json.dumps({
                'apiName': 'VTubeStudioPublicAPI', 'apiVersion': '1.0',
                'requestID': 'r5', 'messageType': 'InjectParameterDataRequest',
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

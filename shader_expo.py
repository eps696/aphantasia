# # CPPNs in GLSL
# taken from https://github.com/wxs/cppn-to-glsl
# Original code was for the NIPS Creativity Workshop submission 'Interactive CPPNs in GLSL'
# modified Mordvintsev et al's CPPN notebook from https://github.com/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/xy2rgb.ipynb
# https://www.apache.org/licenses/LICENSE-2.0

import numpy as np

### Code to convert to GLSL/HLSL

def cppn_to_shader(layers, fn_name='cppn_fn', mode='shadertoy', verbose=False, fix_aspect=True, size=[1., 1.], precision=8):
    """
    Generate shader code out of the list of dicts defining trained CPPN layers 
    mode='vvvv':
        Exports TextureFX shader file for vvvv
    mode='buffer':
        Exports txt file with values for dynamicbuffer input in TextureFX shader for vvvv (and optionally shader itself)
    mode='td':
        Exports code compatible with TouchDesigner: can be dropped into a 'GLSL TOP'
        (see https://docs.derivative.ca/GLSL_TOP). TouchDesigner can be found at http://derivative.ca
    mode='shadertoy':
        Exports code compatible with the ShaderToy editor at http://shadertoy.com
    mode='bookofshaders':
        Exports code compatible with the Book Of Shaders editor here http://editor.thebookofshaders.com/
    """
    
    # Set True to export TFX template for dynamic buffer mode (just once)
    export_tfx = False

    # the xy2rgb cppn's internal size is the output of its first layer (pre-activation)
    # so will just inspect that to figure it out
    n_hidden = layers[0]['weights'].shape[-1]
    if n_hidden % 4 != 0:
        raise ValueError('Currently only support multiples of 4 for hidden layer size')
    modes = {'vvvv', 'buffer', 'td', 'shadertoy', 'bookofshaders'}
    if mode not in modes:
        raise ValueError('Mode {} not one of the supported modes: {}'.format(mode, modes))

    if verbose and precision < 8: print(' .. precision', precision)
    fmt = '%' + '.%df' % precision

    global hlsl; hlsl = None
    
    if mode == 'buffer': 
        global sbW; sbW = []
        buffer = True
    else: buffer = False

    if mode in ['vvvv', 'buffer']:
        hlsl = True
        snippet = """
float2 R:TARGETSIZE;
float4 """
        for i in range(2, len(layers)-2):
            snippet += "in%d_, " % i
        snippet = snippet[:-2] + ';'
        if mode == 'buffer': 
            snippet += '\nStructuredBuffer<float4> sbW;'
        snippet += """
#define mod(x,y) (x - y * floor(x/y))
#define N_HIDDEN {}
float4 {}(float2 uv) {{
    float4 bufA[N_HIDDEN/4];
    float4 bufB[N_HIDDEN/2];
    float4 tmp;
    bufB[0] = float4(uv.x, uv.y, 0., 0.);
""".format(n_hidden, fn_name)
    elif mode == 'td':
        snippet = """
uniform float uIn0;
uniform float uIn1;
uniform float uIn2;
uniform float uIn3;
out vec4 fragColor;
        """
    elif mode == 'shadertoy':
        snippet ="""
#ifdef GL_ES
precision lowp float;
#endif
"""
    elif mode == 'bookofshaders':
        snippet ="""
#ifdef GL_ES
precision lowp float;
#endif
uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
"""
    
    if not mode in ['vvvv', 'buffer']:
        snippet += """
#define N_HIDDEN {}
vec4 bufA[N_HIDDEN/4];
vec4 bufB[N_HIDDEN/2];
vec4 {}(vec2 coordinate, float in0, float in1, float in2, float in3) {{
    vec4 tmp;
    bufB[0] = vec4(coordinate.x, coordinate.y, 0., 0.);
""".format(n_hidden, fn_name)

    def vec(a):
        """Take a Python array of length 4 (or less) and output code for a GLSL vec4 or HLSL float4, possibly zero-padded at the end"""
        global hlsl, sbW
        if len(a) == 4:
            if hlsl is True:
                if 'sbW' in globals(): # check if sbW defined (working with structbuffer input instead of values)
                    for i in range(4):
                        sbW.append(a[i])
                    return 'sbW[%d]' % (len(sbW)//4-1)
                # return 'float4({})'.format(', '.join(str(x) for x in a))
                return 'float4({})'.format(', '.join(fmt % x for x in a))
            else:
                # return 'vec4({})'.format(', '.join(str(x) for x in a))
                return 'vec4({})'.format(', '.join(fmt % x for x in a))
        else:
            assert len(a) < 4 , 'Length must less than 4'
            return vec(np.concatenate([a, [0.]*(4-len(a))]))
    
    def mat(a):
        # Take a numpy matrix of 4 rows and 4 or fewer columns, and output GLSL or HLSL code for a mat4, 
        # possibly with zeros padded in the last columns
        if a.shape[0] < 4:
            m2 = np.vstack([a, [[0.,0.,0.,0.]] * (4 - a.shape[0])])
            return mat(m2)
        assert a.shape[0] == 4, 'Expected a of shape (4,n<=4). Got: {}.'.format(a.shape)
        global hlsl
        if hlsl is True:
            return 'float4x4({})'.format(', '.join(vec(row) for row in a))
        else:
            return 'mat4({})'.format(', '.join(vec(row) for row in a))
    
    for layer_i, layer_dict in enumerate(layers):
        weight = layer_dict['weights']
        bias = layer_dict['bias']
        activation = layer_dict['activation']
        
        _, _, from_size, to_size = weight.shape
        if verbose: print('Processing layer {}. from_size={}, to_size={} .. shape {}'.format(layer_i, from_size, to_size, weight.shape))
        snippet += '\n // layer {} \n'.format(layer_i)
        
        # First, compute the transformation from the last layer into bufA
        for to_index in range(max(1,to_size//4)):
            #Again, the max(1) is important here, because to_size is 3 for the last layer!
            if verbose: print('  generating output {} into bufA'.format(to_index))
            snippet += 'bufA[{}] = {}'.format(to_index, vec(bias[to_index*4:to_index*4+4]))
            if verbose: print('bufA[{}] = {} . . .'.format(to_index, vec(bias[to_index*4:to_index*4+4])))
            for from_index in range(max(1,from_size//4)):
                # the 'max' in the above loop gives us a special case for the first layer, where there are only two inputs.
                if mode in ['vvvv', 'buffer']:
                    snippet += ' + mul(bufB[{}], {})'.format(from_index, mat(weight[0, 0, from_index*4:from_index*4+4, to_index*4:to_index*4+4]))
                    # snippet += ' + mul({}, bufB[{}])'.format(mat(weight[0, 0, from_index*4:from_index*4+4, to_index*4:to_index*4+4]), from_index)
                else:
                    snippet += ' + {} * bufB[{}]'.format(mat(weight[0, 0, from_index*4:from_index*4+4, to_index*4:to_index*4+4]), from_index)
            if mode in ['vvvv', 'buffer'] and layer_i > 1 and layer_i < len(layers)-2:
                suffix = ['x','y','z','w']
                snippet += ' + in{}_.{}'.format(layer_i, suffix[to_index%4])
            else:
                if layer_i == 3:
                    snippet += ' + in{}'.format(to_index%4)
            snippet += ';\n'

        # print('export', layer_i, activation)
        if to_size != 3:
            if verbose: print('  Doing the activation into bufB')
            for to_index in range(to_size//4):
                if activation == 'comp':
                    snippet += 'tmp = atan(bufA[{}]);\n'.format(to_index)
                    snippet += 'bufB[{}] = tmp/0.67;\n'.format(to_index)
                    snippet += 'bufB[{}] = (tmp*tmp) / 0.6;\n'.format(to_index + to_size//4)
                elif activation == 'unbias':
                    snippet += 'tmp = atan(bufA[{}]);\n'.format(to_index)
                    snippet += 'bufB[{}] = tmp/0.67;\n'.format(to_index)
                    snippet += 'bufB[{}] = (tmp*tmp - 0.45) / 0.396;\n'.format(to_index + to_size//4)
                elif activation == 'relu':
                    snippet += 'bufB[{}] = (max(bufA[{}], 0.) - 0.4) / 0.58;\n'.format(to_index, to_index)
                else:
                    raise ValueError('Unknown activation: {}'.format(activation.__name__))
        else:
            if verbose: print('  Sigmoiding the last layer')
            # sigmoid at the last layer
            sigmoider = lambda s: '1. / (1. + exp(-{}))'.format(s)
            if mode in ['vvvv', 'buffer']:
                snippet += '\n return float4(({}).rgb, 1.0);\n'.format(sigmoider('bufA[0]'))
                # snippet += '\n return float4((1. / (1. + exp(-bufA[0]))).xyz, 1.0);\n}'
            else:
                snippet += '\n return vec4(({}).xyz, 1.0);\n'.format(sigmoider('bufA[0]'))
                # snippet += '\n return vec4((1. / (1. + exp(-bufA[0]))).xyz, 1.0);\n}'
    snippet += '}\n'

    if mode in ['vvvv', 'buffer']:
        snippet += """
float4 PS(float4 p:SV_Position, float2 uv:TEXCOORD0): SV_Target {
    uv = 2 * (uv - 0.5);
"""
        if fix_aspect:
            snippet += """
    uv *= R/R.y;
"""
        snippet += """
    return {}(2*uv);
}}
technique10 Process
{{	pass P0 
	{{ SetPixelShader(CompileShader(ps_4_0,PS())); }}
}}
""".format(fn_name)
    elif mode == 'td':
        snippet += """
void main() {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = vUV.xy;
"""
        if fix_aspect:
            snippet += """
    // TODO: don't know how to find the resolution of the GLSL Top output to fix aspect...
"""
        snippet += """
    // Shifted to the form expected by the CPPN
    uv.xy = vec2(1., -1.) * 2. * (uv.xy - vec2(0.5, 0.5));
    uv.y /= {} / {};
    // Output to screen
    fragColor = TDOutputSwizzle({}(uv.xy, uIn0, uIn1, uIn2, uIn3));
}}
        """.format(float(size[0]), float(size[1]), fn_name)
    elif mode == 'shadertoy':
        snippet += """
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    vec2 mouseNorm = (iMouse.xy / iResolution.xy) - vec2(0.5, 0.5);
"""
        if fix_aspect:
            snippet += """
    uv.x *= iResolution.x / iResolution.y;
    uv.x -= ((iResolution.x / iResolution.y) - 1.) /2.;
"""
        snippet += """
    // Shifted to the form expected by the CPPN
    uv = vec2(1., -1.) * 1.5 * (uv - vec2(0.5, 0.5));
    uv.y /= {} / {};
    // Output to screen
    fragColor = {}(uv, 0.23*sin(iTime), 0.32*sin(0.69*iTime), 0.32*sin(0.44*iTime), 0.23*sin(1.23*iTime));
}}
        """.format(float(size[0]), float(size[1]), fn_name)
    elif mode=='bookofshaders':
        snippet += """
void main() {
    vec2 st = gl_FragCoord.xy/u_resolution.xy;
"""
        if fix_aspect:
            snippet += """
    st.x *= u_resolution.x/u_resolution.y;
    st.x -= ((u_resolution.x / u_resolution.y) - 1.) /2.;
"""
        snippet += """
    st = vec2(1., -1.) * 1.5 * (st - vec2(0.5, 0.5));
    st.y /= {} / {};
    gl_FragColor = {}(st, 0.23*sin(u_time), 0.32*sin(0.69*u_time), 0.32*sin(0.44*u_time), 0.23*sin(1.23*u_time));
}}
""".format(float(size[0]), float(size[1]), fn_name)

    if buffer is True:
        # buffer = ','.join('%.8f'%x for x in sbW)
        buffer = ','.join(fmt % x for x in sbW)
        if export_tfx == True:
            with open('CPPN-%d-%d.tfx' % (len(layers)-1, n_hidden), 'w') as f:
                f.write(snippet)
        # print(' total values', len(sbW))
        return buffer
    else:
        return snippet


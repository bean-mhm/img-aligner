#version 450

// push constants
layout(push_constant, std430) uniform pc {
    layout(offset = 0) float img_mul;
    layout(offset = 4) int use_flim;
    layout(offset = 8) int single_channel;
};

// uniforms
layout(binding = 0) uniform sampler2D img;

// output from fragment shader
layout(location = 0) out vec4 out_col;



/*-----------------------------------------------

flim - Filmic Color Transform

Input Color Space:   Linear BT.709 I-D65
Output Color Space:  Linear BT.709 I-D65 / sRGB 2.2 (depends on arguments)

Description:
  Experimental port of flim for GLSL/Shadertoy
  matching flim v1.1.0.

Author:
  Bean (beans_please on Shadertoy)

Original Repo:
  https://github.com/bean-mhm/flim

Original Shader:
  https://www.shadertoy.com/view/dd2yDz

-----------------------------------------------*/

// parameters

const float flim_pre_exposure = 4.3;
const vec3 flim_pre_formation_filter = vec3(1.);
const float flim_pre_formation_filter_strength = 0.;

const float flim_extended_gamut_red_scale = 1.05;
const float flim_extended_gamut_green_scale = 1.12;
const float flim_extended_gamut_blue_scale = 1.045;
const float flim_extended_gamut_red_rot = .5;
const float flim_extended_gamut_green_rot = 2.;
const float flim_extended_gamut_blue_rot = .1;
const float flim_extended_gamut_red_mul = 1.;
const float flim_extended_gamut_green_mul = 1.;
const float flim_extended_gamut_blue_mul = 1.;

const float flim_sigmoid_log2_min = -10.;
const float flim_sigmoid_log2_max = 22.;
const float flim_sigmoid_toe_x = .44;
const float flim_sigmoid_toe_y = .28;
const float flim_sigmoid_shoulder_x = .591;
const float flim_sigmoid_shoulder_y = .779;

const float flim_negative_film_exposure = 6.;
const float flim_negative_film_density = 5.;

const vec3 flim_print_backlight = vec3(1);
const float flim_print_film_exposure = 6.;
const float flim_print_film_density = 27.5;

const float flim_black_point = -1.; // -1 = auto
const vec3 flim_post_formation_filter = vec3(1);
const float flim_post_formation_filter_strength = 0.;
const float flim_midtone_saturation = 1.02;

// color space conversions
// the matrices below represent data in row-major, but GLSL matrices are in
// column-major, so we need to multiply a vec3 by a matrix rather than
// multiplying a matrix by a vec3.

const mat3 mat_bt2020_to_xyz = mat3(
     0.6369580483,  0.1446169036,  0.1688809752,
     0.2627002120,  0.6779980715,  0.0593017165,
     0.0000000000,  0.0280726930,  1.0609850577
);

const mat3 mat_xyz_to_bt2020 = mat3(
     1.7166511880, -0.3556707838, -0.2533662814,
    -0.6666843518,  1.6164812366,  0.0157685458,
     0.0176398574, -0.0427706133,  0.9421031212
);

const mat3 mat_bt709_to_xyz = mat3(
     0.4123907993,  0.3575843394,  0.1804807884,
     0.2126390059,  0.7151686788,  0.0721923154,
     0.0193308187,  0.1191947798,  0.9505321522
);

const mat3 mat_xyz_to_bt709 = mat3(
     3.2409699419, -1.5373831776, -0.4986107603,
    -0.9692436363,  1.8759675015,  0.0415550574,
     0.0556300797, -0.2039769589,  1.0569715142
);

const mat3 mat_dcip3_to_xyz = mat3(
     0.4451698156,  0.2771344092,  0.1722826698,
     0.2094916779,  0.7215952542,  0.0689130679,
     0.0000000000,  0.0470605601,  0.9073553944
);

const mat3 mat_xyz_to_dcip3 = mat3(
     2.7253940305, -1.0180030062, -0.4401631952,
    -0.7951680258,  1.6897320548,  0.0226471906,
     0.0412418914, -0.0876390192,  1.1009293786
);

vec3 oetf_pow(vec3 col, float power)
{
    return pow(col, vec3(1. / power));
}

vec3 eotf_pow(vec3 col, float power)
{
    return pow(col, vec3(power));
}

// flim's utility functions

float flim_wrap(float v, float start, float end)
{
    return start + mod(v - start, end - start);
}

float flim_remap(
    float v,
    float inp_start,
    float inp_end,
    float out_start,
    float out_end
)
{
    return out_start
        + ((out_end - out_start) / (inp_end - inp_start)) * (v - inp_start);
}

float flim_remap_clamp(
    float v,
    float inp_start,
    float inp_end,
    float out_start,
    float out_end
)
{
    float t = clamp((v - inp_start) / (inp_end - inp_start), 0., 1.);
    return out_start + t * (out_end - out_start);
}

float flim_remap01(
    float v,
    float inp_start,
    float inp_end
)
{
    return clamp((v - inp_start) / (inp_end - inp_start), 0., 1.);
}

vec3 flim_blender_rgb_to_hsv(vec3 rgb)
{
    float cmax, cmin, h, s, v, cdelta;
    vec3 c;

    cmax = max(rgb[0], max(rgb[1], rgb[2]));
    cmin = min(rgb[0], min(rgb[1], rgb[2]));
    cdelta = cmax - cmin;

    v = cmax;
    if (cmax != 0.)
    {
        s = cdelta / cmax;
    }
    else
    {
        s = 0.;
        h = 0.;
    }

    if (s == 0.)
    {
        h = 0.;
    }
    else
    {
        c = (vec3(cmax) - rgb.xyz) / cdelta;

        if (rgb.x == cmax)
        {
            h = c[2] - c[1];
        }
        else if (rgb.y == cmax)
        {
            h = 2. + c[0] - c[2];
        }
        else
        {
            h = 4. + c[1] - c[0];
        }

        h /= 6.;

        if (h < 0.)
        {
            h += 1.;
        }
    }

    return vec3(h, s, v);
}

vec3 flim_blender_hsv_to_rgb(vec3 hsv)
{
    float f, p, q, t, h, s, v;
    vec3 rgb;

    h = hsv[0];
    s = hsv[1];
    v = hsv[2];

    if (s == 0.)
    {
        rgb = vec3(v, v, v);
    }
    else
    {
        if (h == 1.)
        {
            h = 0.;
        }

        h *= 6.;
        int i = int(floor(h));
        f = h - float(i);
        rgb = vec3(f, f, f);
        p = v * (1. - s);
        q = v * (1. - (s * f));
        t = v * (1. - (s * (1. - f)));

        if (i == 0)
        {
            rgb = vec3(v, t, p);
        }
        else if (i == 1)
        {
            rgb = vec3(q, v, p);
        }
        else if (i == 2)
        {
            rgb = vec3(p, v, t);
        }
        else if (i == 3)
        {
            rgb = vec3(p, q, v);
        }
        else if (i == 4)
        {
            rgb = vec3(t, p, v);
        }
        else
        {
            rgb = vec3(v, p, q);
        }
    }

    return rgb;
}

vec3 flim_blender_hue_sat(vec3 col, float hue, float sat, float value)
{
    vec3 hsv = flim_blender_rgb_to_hsv(col);

    hsv[0] = fract(hsv[0] + hue + .5);
    hsv[1] = clamp(hsv[1] * sat, 0., 1.);
    hsv[2] = hsv[2] * value;

    return flim_blender_hsv_to_rgb(hsv);
}

float flim_rgb_avg(vec3 col)
{
    return (col.x + col.y + col.z) / 3.;
}

float flim_rgb_sum(vec3 col)
{
    return col.x + col.y + col.z;
}

float flim_rgb_max(vec3 col)
{
    return max(max(col.x, col.y), col.z);
}

float flim_rgb_min(vec3 col)
{
    return min(min(col.x, col.y), col.z);
}

vec3 flim_rgb_uniform_offset(vec3 col, float black_point, float white_point)
{
    float mono = flim_rgb_avg(col);
    float mono2 = flim_remap01(
        mono, black_point / 1000.,
        1. - (white_point / 1000.)
    );
    return col * (mono2 / mono);
}

vec3 flim_rgb_sweep(float hue)
{
    hue = flim_wrap(hue * 360., 0., 360.);

    vec3 col = vec3(1, 0, 0);
    col = mix(col, vec3(1, 1, 0), flim_remap01(hue, 0., 60.));
    col = mix(col, vec3(0, 1, 0), flim_remap01(hue, 60., 120.));
    col = mix(col, vec3(0, 1, 1), flim_remap01(hue, 120., 180.));
    col = mix(col, vec3(0, 0, 1), flim_remap01(hue, 180., 240.));
    col = mix(col, vec3(1, 0, 1), flim_remap01(hue, 240., 300.));
    col = mix(col, vec3(1, 0, 0), flim_remap01(hue, 300., 360.));
    
    return col;
}

vec3 flim_rgb_exposure_sweep_test(vec2 uv0to1)
{
    float hue = 1. - uv0to1.y;
    float exposure = flim_remap(uv0to1.x, 0., 1., -5., 10.);
    return flim_rgb_sweep(hue) * pow(2., exposure);
}

// https://www.desmos.com/calculator/khkztixyeu
float flim_super_sigmoid(
    float v,
    float toe_x,
    float toe_y,
    float shoulder_x,
    float shoulder_y
)
{
    // clip
    v = clamp(v, 0., 1.);
    toe_x = clamp(toe_x, 0., 1.);
    toe_y = clamp(toe_y, 0., 1.);
    shoulder_x = clamp(shoulder_x, 0., 1.);
    shoulder_y = clamp(shoulder_y, 0., 1.);

    // calculate straight line slope
    float slope = (shoulder_y - toe_y) / (shoulder_x - toe_x);

    // toe
    if (v < toe_x)
    {
        float toe_pow = slope * toe_x / toe_y;
        return toe_y * pow(v / toe_x, toe_pow);
    }

    // straight line
    if (v < shoulder_x)
    {
        float intercept = toe_y - (slope * toe_x);
        return slope * v + intercept;
    }

    // shoulder
    float shoulder_pow =
        -slope / (
            ((shoulder_x - 1.) / pow(1. - shoulder_x, 2.))
            * (1. - shoulder_y)
        );
    return
        (1. - pow(1. - (v - shoulder_x) / (1. - shoulder_x), shoulder_pow))
        * (1. - shoulder_y)
        + shoulder_y;
}

float flim_dye_mix_factor(float mono, float max_density)
{
    // log2 and map range
    float offset = pow(2., flim_sigmoid_log2_min);
    float fac = flim_remap01(
        log2(mono + offset),
        flim_sigmoid_log2_min,
        flim_sigmoid_log2_max
    );

    // calculate amount of exposure from 0 to 1
    fac = flim_super_sigmoid(
        fac,
        flim_sigmoid_toe_x,
        flim_sigmoid_toe_y,
        flim_sigmoid_shoulder_x,
        flim_sigmoid_shoulder_y
    );

    // calculate dye density
    fac *= max_density;

    // mix factor
    fac = pow(2., -fac);

    // clip and return
    return clamp(fac, 0., 1.);
}

vec3 flim_rgb_color_layer(
    vec3 col,
    vec3 sensitivity_tone,
    vec3 dye_tone,
    float max_density
)
{
    // normalize
    vec3 sensitivity_tone_norm =
        sensitivity_tone / flim_rgb_sum(sensitivity_tone);
    vec3 dye_tone_norm = dye_tone / flim_rgb_max(dye_tone);

    // dye mix factor
    float mono = dot(col, sensitivity_tone_norm);
    float mix_fac = flim_dye_mix_factor(mono, max_density);

    // dye mixing
    return mix(dye_tone_norm, vec3(1), mix_fac);
}

vec3 flim_rgb_develop(vec3 col, float exposure, float max_density)
{
    // exposure
    col *= pow(2., exposure);

    // blue-sensitive layer
    vec3 result = flim_rgb_color_layer(
        col,
        vec3(0, 0, 1),
        vec3(1, 1, 0),
        max_density
    );

    // green-sensitive layer
    result *= flim_rgb_color_layer(
        col,
        vec3(0, 1, 0),
        vec3(1, 0, 1),
        max_density
    );

    // red-sensitive layer
    result *= flim_rgb_color_layer(
        col,
        vec3(1, 0, 0),
        vec3(0, 1, 1),
        max_density
    );

    return result;
}

vec3 flim_gamut_extension_mat_row(
    float primary_hue,
    float scale,
    float rotate,
    float mul
)
{
    vec3 result = flim_blender_hsv_to_rgb(vec3(
        flim_wrap(primary_hue + (rotate / 360.), 0., 1.),
        1. / scale,
        1.
    ));
    result /= flim_rgb_sum(result);
    result *= mul;
    return result;
}

mat3 flim_gamut_extension_mat(
    float red_scale,
    float green_scale,
    float blue_scale,
    float red_rot,
    float green_rot,
    float blue_rot,
    float red_mul,
    float green_mul,
    float blue_mul
)
{
    mat3 m;
    m[0] = flim_gamut_extension_mat_row(
        0.,
        red_scale,
        red_rot,
        red_mul
    );
    m[1] = flim_gamut_extension_mat_row(
        1. / 3.,
        green_scale,
        green_rot,
        green_mul
    );
    m[2] = flim_gamut_extension_mat_row(
        2. / 3.,
        blue_scale,
        blue_rot,
        blue_mul
    );
    return m;
}

vec3 negative_and_print(vec3 col, vec3 backlight_ext)
{
    // develop negative
    col = flim_rgb_develop(
        col,
        flim_negative_film_exposure,
        flim_negative_film_density
    );

    // backlight
    col *= backlight_ext;

    // develop print
    col = flim_rgb_develop(
        col,
        flim_print_film_exposure,
        flim_print_film_density
    );

    return col;
}

// the flim transform

vec3 flim_transform(vec3 col, float exposure, bool convert_to_srgb)
{
    // eliminate negative values
    col = max(col, 0.);

    // pre-Exposure
    col *= pow(2., flim_pre_exposure + exposure);

    // clip very large values for float precision issues
    col = min(col, 5000.);

    // gamut extension matrix (Linear BT.709)
    mat3 extend_mat = flim_gamut_extension_mat(
        flim_extended_gamut_red_scale,
        flim_extended_gamut_green_scale,
        flim_extended_gamut_blue_scale,
        flim_extended_gamut_red_rot,
        flim_extended_gamut_green_rot,
        flim_extended_gamut_blue_rot,
        flim_extended_gamut_red_mul,
        flim_extended_gamut_green_mul,
        flim_extended_gamut_blue_mul
    );
    mat3 extend_mat_inv = inverse(extend_mat);

    // backlight in the extended gamut
    vec3 backlight_ext = flim_print_backlight * extend_mat;

    // upper limit in the print (highlight cap)
    const float big = 10000000.;
    vec3 white_cap = negative_and_print(vec3(big, big, big), backlight_ext);

    // pre-formation filter
    col = mix(
        col,
        col * flim_pre_formation_filter,
        flim_pre_formation_filter_strength
    );

    // convert to the extended gamut
    col *= extend_mat;

    // negative & print
    col = negative_and_print(col, backlight_ext);

    // convert from the extended gamut
    col *= extend_mat_inv;

    // eliminate negative values
    col = max(col, 0.);

    // white cap
    col /= white_cap;

    // black cap (-1 = auto)
    if (flim_black_point == -1.)
    {
        vec3 black_cap = negative_and_print(vec3(0.), backlight_ext);
        black_cap /= white_cap;
        col = flim_rgb_uniform_offset(
            col,
            flim_rgb_avg(black_cap) * 1000.,
            0.
        );
    }
    else
    {
        col = flim_rgb_uniform_offset(col, flim_black_point, 0.);
    }

    // post-formation filter
    col = mix(
        col,
        col * flim_post_formation_filter,
        flim_post_formation_filter_strength
    );

    // clip
    col = clamp(col, 0., 1.);

    // midtone saturation
    float mono = flim_rgb_avg(col);
    float mix_fac =
        (mono < .5)
        ? flim_remap01(mono, .05, .5)
        : flim_remap01(mono, .95, .5);
    col = mix(
        col,
        flim_blender_hue_sat(col, .5, flim_midtone_saturation, 1.),
        mix_fac
    );

    // clip
    col = clamp(col, 0., 1.);

    // OETF
    if (convert_to_srgb)
    {
        col = oetf_pow(col, 2.2);
    }

    return col;
}

/*____________________ end ____________________*/



void main()
{
    // sample the image pixel-to-pixel so the image will be rendered to the
    // bottom-left corner. also, clamp the UV coordinates.
    vec2 texcoord = gl_FragCoord.xy / vec2(textureSize(img, 0));
    out_col = texture(img, clamp(texcoord, 0., 1.));

    // apply multiplier to the RGB channels
    out_col.rgb *= img_mul;

    // eliminae negative values
    out_col.rgb = max(out_col.rgb, 0.);

    if (single_channel == 1)
    {
        out_col.rgb = vec3(out_col.r);
        out_col.a = 1.;
    }

    // flim
    if (use_flim == 1)
    {
        out_col.rgb = flim_transform(out_col.rgb, 0., false);
    }

    // OETF (Linear BT.709 I-D65 to sRGB 2.2) (doesn't matter if the user's
    // display device uses something else because thanks to ASWF I was not
    // able to import OpenColorIO using CMake after days of agony. you could
    // call it skill issue, I don't care)
    out_col.rgb = pow(out_col.rgb, vec3(1. / 2.2));
    out_col.rgb = clamp(out_col.rgb, 0., 1.);
}

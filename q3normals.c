// Compile with gcc q3normals.c -shared -o q3normals.so
// Taken from the Quake 3 sources licensed under the GNU GPL
#include <math.h>
#include <stdint.h>

// From https://github.com/id-Software/Quake-III-Arena/blob/dbe4ddb10315479fc00086f08e25d968b4b43c49/common/mathlib.h#L43
#define	Q_PI	3.14159265358979323846
// From https://github.com/id-Software/Quake-III-Arena/blob/dbe4ddb10315479fc00086f08e25d968b4b43c49/common/mathlib.h#L45
#define RAD2DEG( a ) ( ( (a) * 180.0f ) / Q_PI )

// From https://github.com/id-Software/Quake-III-Arena/blob/dbe4ddb10315479fc00086f08e25d968b4b43c49/common/cmdlib.h#L56
typedef unsigned char byte;
// From https://github.com/id-Software/Quake-III-Arena/blob/dbe4ddb10315479fc00086f08e25d968b4b43c49/common/mathlib.h#L29
typedef float vec_t;
typedef vec_t vec3_t[3];

// From https://github.com/id-Software/Quake-III-Arena/blob/dbe4ddb10315479fc00086f08e25d968b4b43c49/common/mathlib.c#L44
__attribute__((cdecl)) void NormalToLatLong( const vec3_t normal, byte bytes[2] ) {
	// check for singularities
	if ( normal[0] == 0 && normal[1] == 0 ) {
		if ( normal[2] > 0 ) {
			bytes[0] = 0;
			bytes[1] = 0;		// lat = 0, long = 0
		} else {
			bytes[0] = 128;
			bytes[1] = 0;		// lat = 0, long = 128
		}
	} else {
		int	a, b;

		a = RAD2DEG( atan2( normal[1], normal[0] ) ) * (255.0f / 360.0f );
		a &= 0xff;

		b = RAD2DEG( acos( normal[2] ) ) * ( 255.0f / 360.0f );
		b &= 0xff;

		bytes[0] = b;	// longitude
		bytes[1] = a;	// lattitude
	}
}

// From https://github.com/id-Software/Quake-III-Arena/blob/dbe4ddb10315479fc00086f08e25d968b4b43c49/q3map/misc_model.c#L294
__attribute__((cdecl)) void LatLongToNormal(int16_t latlong, vec3_t xyzout)
{
	float lat, lng;
	lat = ( latlong >> 8 ) & 0xff;
	lng = ( latlong & 0xff );
	lat *= Q_PI/128;
	lng *= Q_PI/128;

	xyzout[0] = cos(lat) * sin(lng);
	xyzout[1] = sin(lat) * sin(lng);
	xyzout[2] = cos(lng);
}

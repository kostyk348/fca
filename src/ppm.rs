//! PPM P6 I/O for the CLI (per-channel 8-bit planes).

/// A 3-plane 8-bit image (deinterleaved PPM P6).
#[derive(Clone, Debug)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub planes: [Vec<u8>; 3], // r, g, b as separate planes
}

/// Read a binary PPM (P6, maxval 255). Comments (`#...`) are supported.
///
/// # Errors
/// I/O, header or truncation problems, described in the message.
pub fn read_ppm(path: &str) -> Result<Image, String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let mut pos = 0usize;
    let token = |raw: &[u8], pos: &mut usize| -> Result<String, String> {
        // skip whitespace and comments
        loop {
            while *pos < raw.len() && raw[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            if *pos < raw.len() && raw[*pos] == b'#' {
                while *pos < raw.len() && raw[*pos] != b'\n' {
                    *pos += 1;
                }
            } else {
                break;
            }
        }
        let start = *pos;
        while *pos < raw.len() && !raw[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if start == *pos {
            return Err("unexpected end of header".into());
        }
        Ok(String::from_utf8_lossy(&raw[start..*pos]).into_owned())
    };

    let magic = token(&raw, &mut pos)?;
    if magic != "P6" {
        return Err(format!("not a PPM P6: {path}"));
    }
    let w: usize = token(&raw, &mut pos)?.parse().map_err(|_| "bad width")?;
    let h: usize = token(&raw, &mut pos)?.parse().map_err(|_| "bad height")?;
    let maxv: usize = token(&raw, &mut pos)?.parse().map_err(|_| "bad maxval")?;
    if maxv != 255 {
        return Err(format!("unsupported maxval {maxv} (only 255)"));
    }
    // single whitespace after maxval
    if pos >= raw.len() || !raw[pos].is_ascii_whitespace() {
        return Err("missing whitespace after maxval".into());
    }
    pos += 1;
    let n = w.checked_mul(h).ok_or("image too large")?;
    if raw.len() < pos + n * 3 {
        return Err("truncated pixel data".into());
    }
    let mut planes: [Vec<u8>; 3] = [vec![0u8; n], vec![0u8; n], vec![0u8; n]];
    for i in 0..n {
        for c in 0..3 {
            planes[c][i] = raw[pos + i * 3 + c];
        }
    }
    Ok(Image { w, h, planes })
}

/// Write a binary PPM (P6).
///
/// # Errors
/// I/O problems, described in the message.
pub fn write_ppm(path: &str, img: &Image) -> Result<(), String> {
    let n = img.w * img.h;
    let mut buf = Vec::with_capacity(16 + n * 3);
    buf.extend_from_slice(format!("P6\n{} {}\n255\n", img.w, img.h).as_bytes());
    for i in 0..n {
        for c in 0..3 {
            buf.push(img.planes[c][i]);
        }
    }
    std::fs::write(path, buf).map_err(|e| format!("cannot write {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_comments() {
        let dir = std::env::temp_dir().join(format!("fca-ppm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.ppm");
        std::fs::write(&p, b"P6\n# a comment\n4 # inline\n2\n255\n").unwrap();
        let mut px = vec![0u8; 4 * 2 * 3];
        for (i, v) in px.iter_mut().enumerate() {
            *v = (i % 256) as u8;
        }
        let mut raw = std::fs::read(&p).unwrap();
        raw.extend_from_slice(&px);
        std::fs::write(&p, &raw).unwrap();

        let img = read_ppm(p.to_str().unwrap()).unwrap();
        assert_eq!(img.w, 4);
        assert_eq!(img.h, 2);
        assert_eq!(img.planes[0][0], px[0]);
        assert_eq!(img.planes[2][7], px[23]);

        let out = dir.join("o.ppm");
        write_ppm(out.to_str().unwrap(), &img).unwrap();
        let img2 = read_ppm(out.to_str().unwrap()).unwrap();
        assert_eq!(img.planes, img2.planes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_bad_files() {
        let dir = std::env::temp_dir().join(format!("fca-ppm-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.ppm");
        std::fs::write(&p, b"P5\n4 2\n255\n").unwrap();
        assert!(
            read_ppm(p.to_str().unwrap()).is_err(),
            "P5 must be rejected"
        );
        std::fs::write(&p, b"P6\n4 2\n255\nshort").unwrap();
        assert!(
            read_ppm(p.to_str().unwrap()).is_err(),
            "truncation must be rejected"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

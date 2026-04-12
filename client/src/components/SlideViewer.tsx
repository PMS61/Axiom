import React, { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import mermaid from "mermaid";
import "katex/dist/katex.min.css";

// ── Common Styling ──────────────────────────────────────────────

const wrapperStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "16px",
  width: "100%",
  height: "100%",
  flex: 1,
};

const titleStyle: React.CSSProperties = {
  fontFamily: "var(--font-playfair)",
  fontSize: "32px",
  lineHeight: "1.2",
  color: "var(--ink)",
  marginBottom: "8px",
  letterSpacing: "-0.5px"
};

const subtitleStyle: React.CSSProperties = {
  fontFamily: "var(--mono)",
  fontSize: "14px",
  color: "var(--muted)",
  textTransform: "uppercase",
};

const textStyle: React.CSSProperties = {
  fontFamily: "var(--mono)",
  fontSize: "16px",
  lineHeight: "1.6",
  color: "var(--ink)",
  whiteSpace: "normal"
};

const bulletsStyle = {
  paddingLeft: "24px",
  margin: "12px 0",
  display: "flex",
  flexDirection: "column",
  gap: "8px",
} as React.CSSProperties;

const preStyle: React.CSSProperties = {
  backgroundColor: "var(--rule)",
  padding: "16px",
  fontFamily: "var(--mono)",
  fontSize: "14px",
  color: "var(--bg)",
  overflowX: "auto",
  marginTop: "16px",
  marginBottom: "16px",
};

// ── Helper Components ─────────────────────────────────────────

const MarkdownText = ({ children }: { children: string }) => {
  if (!children) return null;
  return (
    <div style={textStyle}>
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[rehypeKatex]}
        components={{
          p: ({ node, ...props }) => <p style={{ margin: 0, paddingBottom: "8px" }} {...props} />,
          li: ({ node, ...props }) => <li style={{ marginBottom: "4px" }} {...props} />,
          h1: ({ node, ...props }) => <h1 style={titleStyle} {...props} />,
          h2: ({ node, ...props }) => <h2 style={{ ...titleStyle, fontSize: "24px" }} {...props} />,
          h3: ({ node, ...props }) => <h3 style={{ ...titleStyle, fontSize: "20px" }} {...props} />,
          a: ({ node, ...props }) => <a style={{ textDecoration: "underline", color: "var(--ink)" }} {...props} />,
          strong: ({ node, ...props }) => <strong style={{ fontWeight: "bold" }} {...props} />,
          em: ({ node, ...props }) => <em style={{ fontStyle: "italic" }} {...props} />
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
};

const MermaidDiagram = ({ chart }: { chart: string }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current && chart) {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'base',
        fontFamily: 'var(--mono)',
        themeVariables: {
          primaryColor: '#1E1E1E',      // var(--card-bg)
          primaryTextColor: '#FAF7F2',  // var(--ink)
          primaryBorderColor: '#333333', // var(--rule)
          lineColor: '#A3A3A3',         // var(--muted)
          secondaryColor: '#141414',    // var(--bg)
          tertiaryColor: '#141414',
          nodeBorder: '#333333',
          clusterBkg: 'transparent',
          edgeLabelBackground: '#1E1E1E'
        }
      });
      const id = `mermaid-${Math.random().toString(36).substring(2, 9)}`;
      try {
        mermaid.render(id, chart)
          .then((result) => {
            if (containerRef.current) {
              let svg = result.svg;

              if (!svg.includes('preserveAspectRatio')) {
                svg = svg.replace('<svg ', '<svg preserveAspectRatio="xMidYMid meet" ');
              }

              containerRef.current.innerHTML = svg;

              const svgElement = containerRef.current.querySelector('svg');
              if (svgElement) {
                // Tighten oversized Mermaid viewBoxes so the graph uses available slide space.
                const graphRoot = svgElement.querySelector('g');
                if (graphRoot && typeof graphRoot.getBBox === 'function') {
                  const bbox = graphRoot.getBBox();
                  if (bbox.width > 0 && bbox.height > 0) {
                    const pad = 24;
                    svgElement.setAttribute(
                      'viewBox',
                      `${bbox.x - pad} ${bbox.y - pad} ${bbox.width + pad * 2} ${bbox.height + pad * 2}`,
                    );
                  }
                }

                svgElement.removeAttribute('width');
                svgElement.removeAttribute('height');
                svgElement.style.width = '100%';
                svgElement.style.height = 'auto';
                svgElement.style.display = 'block';
                svgElement.style.margin = '0 auto';
              }
            }
          })
          .catch(e => {
            console.error("Mermaid error:", e);
            if (containerRef.current) {
              containerRef.current.innerHTML = `<pre style="padding: 16px; color: var(--vermillion); background: var(--rule); font-family: var(--mono); font-size: 14px; overflow-x: auto;">${e.message}</pre>`;
            }
          });
      } catch (err: any) {
        if (containerRef.current) {
          containerRef.current.innerHTML = `<pre style="padding: 16px; color: var(--vermillion); background: var(--rule); font-family: var(--mono); font-size: 14px; overflow-x: auto;">${err?.message}</pre>`;
        }
      }
    }
  }, [chart]);

  return (
    <div style={{
      width: "100%",
      flex: 1,
      minHeight: 0,
      display: "flex",
      justifyContent: "center",
      alignItems: "flex-start",
      overflow: "auto",
      backgroundColor: "transparent",
      padding: "8px 0"
    }}>
      <div
        ref={containerRef}
        style={{
          width: "100%",
          minHeight: "280px",
          display: "block",
          overflow: "visible",
        }}
      />
    </div>
  );
};


// ── Slide Viewer ─────────────────────────────────────────────

export default function SlideViewer({ slide }: { slide: any }) {
  if (!slide) {
    return (
      <div style={{ ...wrapperStyle, alignItems: "center", justifyContent: "center", minHeight: "200px" }}>
        <div style={textStyle}>No content available for this slide.</div>
      </div>
    );
  }

  let type = slide.type;
  if (!type) {
    if (slide.diagram !== undefined) type = "mermaid";
    else if (slide.imageUrl !== undefined && slide.content !== undefined) type = "content-and-image";
    else if (slide.imageUrl !== undefined) type = "image";
    else if (slide.code !== undefined && slide.content !== undefined) type = "content-and-code";
    else if (slide.code !== undefined) type = "code";
    else if (slide.headers !== undefined && slide.rows !== undefined) type = "table";
    else if (slide.quote !== undefined) type = "quote";
    else if (slide.items !== undefined) type = "list";
    else type = "content";
  }

  // Remove scripts from rendering payload to prevent leaks.
  const safeSlide = { ...slide };
  delete safeSlide.script;

  switch (type) {
    case "title":
      return (
        <div style={{ ...wrapperStyle, alignItems: "center", justifyContent: "center", minHeight: "300px", textAlign: "center" }}>
          {slide.title && <h2 style={{ ...titleStyle, fontSize: "48px" }}>{slide.title}</h2>}
          {slide.subtitle && <div style={{ ...subtitleStyle, fontSize: "18px", color: "var(--ink)", width: "100%" }}><MarkdownText>{slide.subtitle}</MarkdownText></div>}
          <div style={{ marginTop: "40px" }}>
            {slide.author && <div style={textStyle}>BY {slide.author}</div>}
            {slide.date && <div style={{ ...textStyle, color: "var(--muted)", fontSize: "12px", marginTop: "4px" }}>{slide.date}</div>}
          </div>
        </div>
      );

    case "section":
      return (
        <div style={{ ...wrapperStyle, alignItems: "center", justifyContent: "center", minHeight: "250px", textAlign: "center" }}>
          {slide.title && <h3 style={{ ...titleStyle, fontSize: "40px" }}>{slide.title}</h3>}
          {slide.description && <div style={{ maxWidth: "80%" }}><MarkdownText>{slide.description}</MarkdownText></div>}
        </div>
      );

    case "content":
    case "conclusion":
      return (
        <div style={wrapperStyle}>
          {slide.title && <h3 style={titleStyle}>{slide.title}</h3>}
          {slide.content && <MarkdownText>{slide.content}</MarkdownText>}
          {slide.bullets && slide.bullets.length > 0 && (
            <ul style={bulletsStyle}>
              {slide.bullets.map((b: string, i: number) => (
                <li key={i}><MarkdownText>{b}</MarkdownText></li>
              ))}
            </ul>
          )}
        </div>
      );

    case "image":
      return (
        <div style={wrapperStyle}>
          {slide.title && <h3 style={titleStyle}>{slide.title}</h3>}
          <div style={{ width: "100%", backgroundColor: "var(--rule)", aspectRatio: "16/9", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
            {slide.imageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={slide.imageUrl} alt={slide.caption || "Slide Image"} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            ) : (
              <span className="meta-text" style={{ color: "var(--bg)" }}>[IMAGE MISSING]</span>
            )}
          </div>
          {slide.caption && <div style={{ ...subtitleStyle, textAlign: "center", marginTop: "8px" }}><MarkdownText>{slide.caption}</MarkdownText></div>}
        </div>
      );

    case "content-and-image":
      return (
        <div style={wrapperStyle}>
          {slide.title && <h3 style={titleStyle}>{slide.title}</h3>}
          <div style={{ display: "flex", gap: "24px", alignItems: "flex-start" }}>
            <div style={{ flex: 1 }}>
              {slide.content && <MarkdownText>{slide.content}</MarkdownText>}
              {slide.bullets && slide.bullets.length > 0 && (
                <ul style={bulletsStyle}>
                  {slide.bullets.map((b: string, i: number) => (
                    <li key={i}><MarkdownText>{b}</MarkdownText></li>
                  ))}
                </ul>
              )}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ width: "100%", backgroundColor: "var(--rule)", aspectRatio: "4/3", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
                {slide.imageUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={slide.imageUrl} alt={slide.imageCaption || "Slide Image"} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                ) : (
                  <span className="meta-text" style={{ color: "var(--bg)" }}>[IMAGE MISSING]</span>
                )}
              </div>
              {slide.imageCaption && <div style={{ ...subtitleStyle, textAlign: "center", marginTop: "8px" }}><MarkdownText>{slide.imageCaption}</MarkdownText></div>}
            </div>
          </div>
        </div>
      );

    case "code":
      return (
        <div style={wrapperStyle}>
          {slide.title && <h3 style={titleStyle}>{slide.title}</h3>}
          {slide.language && <div style={{ ...subtitleStyle, textTransform: "uppercase" }}>LANGUAGE: {slide.language}</div>}
          {slide.code && (
            <pre style={preStyle}>
              <code>{slide.code}</code>
            </pre>
          )}
          {slide.explanation && <MarkdownText>{slide.explanation}</MarkdownText>}
        </div>
      );

    case "content-and-code":
      return (
        <div style={wrapperStyle}>
          {slide.title && <h3 style={titleStyle}>{slide.title}</h3>}
          {slide.content && <MarkdownText>{slide.content}</MarkdownText>}
          {slide.language && <div style={{ ...subtitleStyle, textTransform: "uppercase", marginTop: "16px" }}>LANGUAGE: {slide.language}</div>}
          {slide.code && (
            <pre style={preStyle}>
              <code>{slide.code}</code>
            </pre>
          )}
        </div>
      );

    case "list":
      return (
        <div style={wrapperStyle}>
          {slide.title && <h3 style={titleStyle}>{slide.title}</h3>}
          {slide.items && slide.items.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginTop: "8px" }}>
              {slide.items.map((item: any, i: number) => (
                <div key={i} style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
                  <div style={{ ...textStyle, color: item.checked ? "var(--ink)" : "var(--muted)", paddingTop: "2px" }}>
                    {item.checked ? "[x]" : "[ ]"}
                  </div>
                  <div><MarkdownText>{item.text}</MarkdownText></div>
                </div>
              ))}
            </div>
          )}
        </div>
      );

    case "table":
      return (
        <div style={wrapperStyle}>
          {slide.title && <h3 style={titleStyle}>{slide.title}</h3>}
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "16px" }}>
              <thead>
                <tr>
                  {slide.headers?.map((h: string, i: number) => (
                    <th key={i} style={{ ...subtitleStyle, color: "var(--ink)", borderBottom: "1px solid var(--ink)", padding: "12px", textAlign: "left" }}>
                      <MarkdownText>{h}</MarkdownText>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {slide.rows?.map((row: string[], i: number) => (
                  <tr key={i}>
                    {row.map((cell: string, j: number) => (
                      <td key={j} style={{ ...textStyle, borderBottom: "0.5px solid var(--rule)", padding: "12px" }}>
                        <MarkdownText>{cell}</MarkdownText>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );

    case "quote":
      return (
        <div style={{ ...wrapperStyle, alignItems: "center", justifyContent: "center", padding: "40px" }}>
          <div style={{ ...titleStyle, fontSize: "40px", fontStyle: "italic", textAlign: "center" }}>
            "<MarkdownText>{slide.quote}</MarkdownText>"
          </div>
          <div style={{ ...subtitleStyle, marginTop: "16px", color: "var(--ink)" }}>— {slide.author}</div>
          {slide.context && <div style={{ maxWidth: "80%", textAlign: "center", marginTop: "16px" }}><MarkdownText>{slide.context}</MarkdownText></div>}
        </div>
      );

    case "mermaid":
      return (
        <div style={wrapperStyle}>
          {slide.title && <h3 style={titleStyle}>{slide.title}</h3>}
          <div style={{ ...subtitleStyle, marginBottom: "8px" }}>Mermaid Diagram</div>
          <MermaidDiagram chart={slide.diagram} />
          {slide.caption && <div style={{ textAlign: "center", marginTop: "8px", fontStyle: "italic" }}><MarkdownText>{slide.caption}</MarkdownText></div>}
        </div>
      );

    default:
      return (
        <div style={wrapperStyle}>
          {safeSlide.title && <h3 style={titleStyle}>{safeSlide.title}</h3>}
          <pre style={{ ...textStyle, fontSize: "12px" }}>
            {JSON.stringify(safeSlide, null, 2)}
          </pre>
        </div>
      );
  }
}

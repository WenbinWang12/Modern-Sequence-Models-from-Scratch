const renderMermaid = () => {
  if (!window.mermaid) return;
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    themeVariables: {
      fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
      primaryColor: "#dceeff",
      primaryTextColor: "#172033",
      primaryBorderColor: "#59708a",
      lineColor: "#59708a",
      secondaryColor: "#ddf7ee",
      tertiaryColor: "#fff0c2",
    },
  });
  window.mermaid.run({ querySelector: ".mermaid" });
};

if (typeof document$ !== "undefined") {
  document$.subscribe(renderMermaid);
} else {
  document.addEventListener("DOMContentLoaded", renderMermaid);
}

// 让顶部的语言选择器跳到“当前页面”的另一种语言版本；没有对应页面时保持默认链接。
(function () {
  function counterpart(path, target) {
    var rules = [
      [/^technical_report\/$/, "technical_report_en/", /^technical_report_en\/$/, "technical_report/"],
      [/^reports\/benchmark_protocol\/$/, "reports/benchmark_protocol_en/", /^reports\/benchmark_protocol_en\/$/, "reports/benchmark_protocol/"],
      [/^book\/(?!en\/)(.+)$/, "book/en/$1", /^book\/en\/(.+)$/, "book/$1"],
      [/^frontier\/(?!en\/)(.*)$/, "frontier/en/$1", /^frontier\/en\/(.*)$/, "frontier/$1"],
      [/^$/, "book/en/", /^book\/en\/$/, ""]
    ];
    for (var i = 0; i < rules.length; i++) {
      var rule = rules[i];
      if (target === "en" && rule[0].test(path)) return path.replace(rule[0], rule[1]);
      if (target === "zh" && rule[2].test(path)) return path.replace(rule[2], rule[3]);
    }
    return null;
  }

  document$.subscribe(function () {
    var scope = typeof __md_scope !== "undefined" ? __md_scope : new URL(".", location);
    var path = location.pathname.slice(scope.pathname.length);
    document.querySelectorAll("a.md-select__link[hreflang]").forEach(function (link) {
      var language = link.getAttribute("hreflang");
      var other = counterpart(path, language);
      if (other === null) {
        // 当前页面已经是这种语言：选择它时留在本页，而不是跳回首页。
        if (counterpart(path, language === "en" ? "zh" : "en") !== null) link.href = location.href;
        return;
      }
      var url = new URL(other, scope);
      fetch(url, { method: "HEAD" })
        .then(function (response) {
          if (response.ok) link.href = url.href;
        })
        .catch(function () {});
    });
  });
})();

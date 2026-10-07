package com.itsolution.mrpscanagent;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.DownloadManager;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.CookieManager;
import android.webkit.JsResult;
import android.webkit.PermissionRequest;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.URLUtil;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * MRPscan Agent: the question-bank console (SERVER_URL/crm) as a phone app.
 *
 * The page is not changed; the app only supplies what a browser would otherwise give
 * it: the microphone for the Record buttons, file downloads and alert/confirm
 * dialogs. The console's own buttons still lead to its other pages.
 */
public class MainActivity extends Activity {
    private static final int REQ_MIC = 1;
    private static final Pattern FILENAME =
            Pattern.compile("filename\\*?=(?:UTF-8'')?\"?([^\";]+)\"?", Pattern.CASE_INSENSITIVE);

    private final List<PermissionRequest> waitingForMic = new ArrayList<>();
    private WebView web;
    private String failure;    // why the page did not load; null = it did
    private String retryUrl;   // the page to try again
    private String lastGood;   // the last page that did load
    private int progress = 100;
    private boolean askingForMic;
    private boolean stopped;
    private String serverHost;

    private FrameLayout frame;
    private View offline;
    private TextView offlineDetail;
    private ProgressBar progressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        serverHost = Uri.parse(BuildConfig.SERVER_URL).getHost();
        WebView.setWebContentsDebuggingEnabled(BuildConfig.WEB_DEBUG);

        frame = findViewById(R.id.web_frame);
        offline = findViewById(R.id.offline);
        offlineDetail = findViewById(R.id.offline_detail);
        progressBar = findViewById(R.id.progress);
        findViewById(R.id.offline_retry).setOnClickListener(v -> retry());

        CookieManager.getInstance().setAcceptCookie(true);
        createWeb();
    }

    private static String home() {
        return BuildConfig.SERVER_URL + "/crm";
    }

    @SuppressLint("SetJavaScriptEnabled")
    private void createWeb() {
        web = new WebView(this);
        web.setBackgroundColor(0xFFF6F7F9);   // the console's grey, until it paints
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setTextZoom(100);                    // the page sizes its own text, as on the web
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setUserAgentString(s.getUserAgentString() + " MRPscanAgentApp/" + BuildConfig.VERSION_NAME);
        web.setWebViewClient(new Client());
        web.setWebChromeClient(new Chrome());
        web.setDownloadListener((url, userAgent, disposition, mimeType, length) ->
                download(url, userAgent, disposition, mimeType));
        frame.addView(web, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        progress = 0;
        web.loadUrl(home());
    }

    /** The offline cover and the loading bar. */
    private void showState() {
        offline.setVisibility(failure != null ? View.VISIBLE : View.GONE);
        if (failure != null) offlineDetail.setText(getString(R.string.offline_detail, failure));
        progressBar.setProgress(progress);
        progressBar.setVisibility(progress < 100 && failure == null ? View.VISIBLE : View.GONE);
    }

    private void retry() {
        String url = retryUrl != null ? retryUrl : (lastGood != null ? lastGood : home());
        failure = null;
        showState();
        if (web == null) createWeb();
        else web.loadUrl(url);
    }

    private void fail(WebResourceRequest request, String why) {
        failure = why;
        // A failed form post (signing in) is not repeated by itself; go back to the page it came from.
        retryUrl = "GET".equalsIgnoreCase(request.getMethod()) ? request.getUrl().toString() : null;
        showState();
    }

    /** The page's process died (usually Android reclaiming memory): drop it instead of the app. */
    private void dropWeb(WebView dead) {
        if (web != dead) return;
        frame.removeView(dead);
        dead.destroy();
        web = null;
        if (!stopped) createWeb();
    }

    private boolean isOurs(Uri uri) {
        if (uri == null || serverHost == null) return false;
        String scheme = uri.getScheme();
        return ("https".equalsIgnoreCase(scheme) || "http".equalsIgnoreCase(scheme))
                && serverHost.equalsIgnoreCase(uri.getHost());
    }

    private static boolean isHome(String url) {
        String path = url == null ? null : Uri.parse(url).getPath();
        return "/crm".equals(path) || "/crm/".equals(path);
    }

    private void openOutside(Uri uri) {
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, uri).addCategory(Intent.CATEGORY_BROWSABLE));
        } catch (ActivityNotFoundException e) {
            toast(getString(R.string.no_app_for_link));
        }
    }

    // ── microphone (the Record buttons) ──────────────────────────────────────

    private void askForMic(PermissionRequest request) {
        boolean wantsMic = false;
        for (String resource : request.getResources()) {
            if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(resource)) wantsMic = true;
        }
        // Only the agent's own pages, and only the microphone: no page here asks for anything else.
        if (!wantsMic || !isOurs(request.getOrigin())) {
            request.deny();
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            request.grant(new String[]{PermissionRequest.RESOURCE_AUDIO_CAPTURE});
            return;
        }
        // A second Record tap while Android's prompt is up waits for the same answer.
        waitingForMic.add(request);
        if (!askingForMic) {
            askingForMic = true;
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQ_MIC);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode != REQ_MIC) return;
        askingForMic = false;
        boolean answered = results.length > 0;
        boolean granted = answered && results[0] == PackageManager.PERMISSION_GRANTED;
        for (PermissionRequest request : waitingForMic) {
            if (granted) request.grant(new String[]{PermissionRequest.RESOURCE_AUDIO_CAPTURE});
            else request.deny();
        }
        waitingForMic.clear();
        if (granted || !answered) return;
        if (shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO)) {
            toast(getString(R.string.mic_needed));
        } else {
            // Refused for good ("Don't allow" twice): only Settings can turn it back on.
            new AlertDialog.Builder(this)
                    .setTitle(R.string.mic_settings_title)
                    .setMessage(getString(R.string.mic_settings_body, getString(R.string.app_name)))
                    .setPositiveButton(R.string.open_settings, (d, w) -> openAppSettings())
                    .setNegativeButton(R.string.not_now, null)
                    .show();
        }
    }

    private void openAppSettings() {
        try {
            startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    Uri.fromParts("package", getPackageName(), null)));
        } catch (ActivityNotFoundException ignored) {
        }
    }

    // ── downloads (the bank's Excel, variants JSON and voice-cache zip) ──────

    private void download(String url, String userAgent, String disposition, String mimeType) {
        Uri uri = Uri.parse(url);
        if (!isOurs(uri)) {
            openOutside(uri);
            return;
        }
        String name = fileName(url, disposition, mimeType);
        try {
            DownloadManager.Request req = new DownloadManager.Request(uri);
            // The downloads sit behind the console's sign-in, which lives in the WebView's cookies.
            String cookies = CookieManager.getInstance().getCookie(url);
            if (cookies != null) req.addRequestHeader("Cookie", cookies);
            if (userAgent != null) req.addRequestHeader("User-Agent", userAgent);
            if (mimeType != null && !mimeType.isEmpty()) req.setMimeType(mimeType);
            req.setTitle(name);
            req.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                req.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, name);
            } else {
                // Before Android 10 the shared Downloads folder needs a storage permission; the app's own does not.
                req.setDestinationInExternalFilesDir(this, Environment.DIRECTORY_DOWNLOADS, name);
            }
            ((DownloadManager) getSystemService(DOWNLOAD_SERVICE)).enqueue(req);
            toast(getString(R.string.downloading, name));
        } catch (Exception e) {
            toast(getString(R.string.download_failed, String.valueOf(e.getMessage())));
        }
    }

    static String fileName(String url, String disposition, String mimeType) {
        if (disposition != null) {
            Matcher m = FILENAME.matcher(disposition);
            if (m.find()) {
                String name = Uri.decode(m.group(1)).replaceAll("[\\\\/:*?\"<>|]", "_").trim();
                if (!name.isEmpty()) return name;
            }
        }
        return URLUtil.guessFileName(url, disposition, mimeType);
    }

    // ── page dialogs ────────────────────────────────────────────────────────

    private boolean showJsDialog(String message, JsResult result, boolean confirm) {
        if (isFinishing() || isDestroyed()) {
            result.cancel();
            return true;
        }
        AlertDialog.Builder b = new AlertDialog.Builder(this)
                .setMessage(message)
                .setPositiveButton(android.R.string.ok, (d, w) -> result.confirm())
                .setOnCancelListener(d -> result.cancel());
        if (confirm) b.setNegativeButton(android.R.string.cancel, (d, w) -> result.cancel());
        b.show();
        return true;
    }

    private void toast(String text) {
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show();
    }

    // ── WebView callbacks ───────────────────────────────────────────────────

    private class Client extends WebViewClient {
        @Override
        public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
            if (!request.isForMainFrame()) return false;
            Uri uri = request.getUrl();
            String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
            if (isOurs(uri) || scheme.equals("about") || scheme.equals("data") || scheme.equals("blob")) return false;
            openOutside(uri);   // anything else belongs to the browser, the dialer, mail…
            return true;
        }

        @Override
        public void onPageStarted(WebView view, String url, Bitmap favicon) {
            if (view != web) return;
            failure = null;
            retryUrl = null;
            showState();
        }

        @Override
        public void onPageFinished(WebView view, String url) {
            if (view != web) return;
            if (failure == null) lastGood = url;
            // The console's first page is the bottom of the history: Back from there leaves the app.
            if (isHome(url)) view.clearHistory();
            CookieManager.getInstance().flush();
        }

        @Override
        public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
            if (view != web) return;
            String why = String.valueOf(error.getDescription());
            // ERR_ABORTED is a load cut short on purpose (a download, or a newer navigation).
            if (request.isForMainFrame() && !why.contains("ERR_ABORTED")) fail(request, why);
        }

        @Override
        public void onReceivedHttpError(WebView view, WebResourceRequest request, WebResourceResponse response) {
            if (view != web) return;
            if (request.isForMainFrame() && response.getStatusCode() >= 500) {
                fail(request, "The agent server answered " + response.getStatusCode() + ".");
            }
        }

        @Override
        public boolean onRenderProcessGone(WebView view, RenderProcessGoneDetail detail) {
            dropWeb(view);
            return true;
        }
    }

    private class Chrome extends WebChromeClient {
        @Override
        public void onProgressChanged(WebView view, int newProgress) {
            if (view != web) return;
            progress = newProgress;
            showState();
        }

        @Override
        public void onPermissionRequest(PermissionRequest request) {
            runOnUiThread(() -> askForMic(request));
        }

        @Override
        public void onPermissionRequestCanceled(PermissionRequest request) {
            waitingForMic.remove(request);
        }

        @Override
        public boolean onJsAlert(WebView view, String url, String message, JsResult result) {
            return showJsDialog(message, result, false);
        }

        @Override
        public boolean onJsConfirm(WebView view, String url, String message, JsResult result) {
            return showJsDialog(message, result, true);
        }
    }

    // ── activity ────────────────────────────────────────────────────────────

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onStart() {
        super.onStart();
        stopped = false;
        if (web == null) createWeb();
        web.resumeTimers();
        web.onResume();
    }

    @Override
    protected void onStop() {
        stopped = true;
        CookieManager.getInstance().flush();
        if (web != null) {        // nothing runs while the app is in the background
            web.onPause();
            web.pauseTimers();
        }
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        for (PermissionRequest request : waitingForMic) request.deny();
        waitingForMic.clear();
        if (web != null) {
            frame.removeView(web);
            web.destroy();
            web = null;
        }
        super.onDestroy();
    }
}

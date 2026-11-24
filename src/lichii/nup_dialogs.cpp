// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "lichii/nup_dialogs.h"

#include <QBuffer>
#include <QButtonGroup>
#include <QCheckBox>
#include <QCloseEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QImageReader>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QProgressBar>
#include <QRadioButton>
#include <QScrollArea>
#include <QSlider>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <array>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "lichii/spotlight/menu.h"
#include "lichii/spotlight/theme.h"
#include "lichii/spotlight/widgets.h"
#include "lichii/util/icon_search.h"

namespace {

constexpr int IconSize = 256;
constexpr int ColumnWidth = 168;
constexpr int CardPadding = 24;

// Scoped to the card, so message boxes and file dialogs opened from a card keep their own style.
constexpr auto CardStyle = R"(
QWidget#archiveCard, QWidget#archiveCard QWidget { background: transparent; }
QWidget#archiveCard QLineEdit {
    background-color: rgba(255, 255, 255, 15);
    border: 1px solid rgba(255, 255, 255, 31);
    border-radius: 12px;
    min-height: 38px;
    padding: 0 13px;
    font-family: "Lexend";
    font-size: 14px;
    font-weight: 500;
    color: #ffffff;
}
QWidget#archiveCard QLineEdit:hover { border-color: rgba(255, 255, 255, 61); }
QWidget#archiveCard QLineEdit:focus { border-color: #ff6b95; }
QWidget#archiveCard QLineEdit:disabled {
    color: rgba(255, 255, 255, 92);
    border-color: rgba(255, 255, 255, 20);
}
QWidget#archiveCard QProgressBar {
    background-color: rgba(255, 255, 255, 46);
    border: 0;
    border-radius: 4px;
    min-height: 8px;
    max-height: 8px;
}
QWidget#archiveCard QProgressBar::chunk {
    background-color: #ff6b95;
    border-radius: 4px;
}
)";

QString ToQString(const std::filesystem::path& path) {
    return QString::fromStdString(Common::FS::PathToUTF8String(path));
}

std::filesystem::path FromQString(const QString& path) {
    const auto utf8 = path.toStdString();
    return std::filesystem::path{std::u8string{utf8.begin(), utf8.end()}};
}

QString HumanSize(u64 bytes) {
    constexpr double GiB = 1024.0 * 1024 * 1024;
    if (bytes >= GiB) {
        return QObject::tr("%1 GB").arg(bytes / GiB, 0, 'f', 1);
    }
    if (bytes >= (u64{1} << 20)) {
        return QObject::tr("%1 MB").arg(bytes >> 20);
    }
    return QObject::tr("%1 KB").arg(std::max<u64>(1, bytes >> 10));
}

QPixmap PixmapFromBytes(const std::vector<u8>& bytes) {
    QPixmap pixmap;
    pixmap.loadFromData(bytes.data(), static_cast<uint>(bytes.size()));
    return pixmap;
}

QPixmap RoundedPreview(const QPixmap& source, int size, qreal radius) {
    const qreal ratio = 2.0;
    QPixmap out(QSize(size, size) * ratio);
    out.setDevicePixelRatio(ratio);
    out.fill(Qt::transparent);
    QPainter painter(&out);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    QPainterPath clip;
    clip.addRoundedRect(QRectF(0, 0, size, size), radius, radius);
    painter.setClipPath(clip);
    painter.drawPixmap(QRectF(0, 0, size, size), source, source.rect());
    return out;
}

QImage ReadImage(const QString& path) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    return reader.read();
}

QRect CenterSquare(const QImage& image) {
    const int side = std::min(image.width(), image.height());
    return QRect((image.width() - side) / 2, (image.height() - side) / 2, side, side);
}

QLabel* TextLabel(const QString& text, int size, int weight, const QColor& color) {
    auto* label = new QLabel(text);
    label->setFont(Spotlight::Font(size, weight));
    Spotlight::SetTextColor(label, color);
    return label;
}

struct LocalizedLanguage {
    const char* tag;
    const char* name;
};

// The languages the console stores a name for; "en-us" is the Name field itself, and tags not
// listed here are kept as stored.
constexpr std::array<LocalizedLanguage, 15> LocalizedLanguages{{
    {"en-gb", QT_TRANSLATE_NOOP("ArchiveEditDialog", "English (UK)")},
    {"ja", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Japanese")},
    {"fr", QT_TRANSLATE_NOOP("ArchiveEditDialog", "French")},
    {"de", QT_TRANSLATE_NOOP("ArchiveEditDialog", "German")},
    {"es-419", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Spanish (Latin America)")},
    {"es", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Spanish")},
    {"it", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Italian")},
    {"nl", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Dutch")},
    {"fr-ca", QT_TRANSLATE_NOOP("ArchiveEditDialog", "French (Canada)")},
    {"pt", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Portuguese")},
    {"ru", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Russian")},
    {"ko", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Korean")},
    {"zh-hant", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Chinese (Traditional)")},
    {"zh-hans", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Chinese (Simplified)")},
    {"pt-br", QT_TRANSLATE_NOOP("ArchiveEditDialog", "Portuguese (Brazil)")},
}};

bool IsEditableLanguage(const std::string& tag) {
    return std::ranges::any_of(LocalizedLanguages,
                               [&](const auto& language) { return tag == language.tag; });
}

// The first dropped local file that Qt can read as an image.
QString DroppedImage(const QMimeData* mime) {
    if (mime == nullptr) {
        return {};
    }
    for (const auto& url : mime->urls()) {
        if (url.isLocalFile() && !QImageReader(url.toLocalFile()).format().isEmpty()) {
            return url.toLocalFile();
        }
    }
    return {};
}

// A rounded box inside the card.
class Box final : public QWidget {
public:
    Box(QColor fill_, QColor stroke_, int radius_, QWidget* parent = nullptr)
        : QWidget(parent), fill{fill_}, stroke{stroke_}, radius{radius_} {}

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(QPen(stroke, 1));
        painter.setBrush(fill);
        painter.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5), radius, radius);
    }

private:
    QColor fill;
    QColor stroke;
    int radius;
};

// One line of text cut in the middle when it does not fit, such as a path.
class ElidedLabel final : public QWidget {
public:
    ElidedLabel(const QFont& font, QColor color_, QWidget* parent = nullptr)
        : QWidget(parent), color{color_} {
        setFont(font);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    }

    void SetText(const QString& text_) {
        text = text_;
        setToolTip(text);
        update();
    }

    QSize sizeHint() const override {
        return {80, fontMetrics().height()};
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setPen(color);
        painter.drawText(rect(), Qt::AlignVCenter | Qt::AlignLeft,
                         fontMetrics().elidedText(text, Qt::ElideMiddle, width()));
    }

private:
    QString text;
    QColor color;
};

// A button without a border: an optional icon and text, filled or bare, optionally with a caret
// after the text that marks a menu.
class FlatButton final : public QAbstractButton {
public:
    FlatButton(const QString& text, QString icon_, bool filled_, int height,
               QWidget* parent = nullptr)
        : QAbstractButton(parent), icon{std::move(icon_)}, filled{filled_}, button_height{height} {
        setText(text);
        setCursor(Qt::PointingHandCursor);
        setFont(Spotlight::Font(height >= 34 ? 13 : 12, 600));
    }

    void SetCaret(bool caret_) {
        caret = caret_;
        updateGeometry();
        update();
    }

    QSize sizeHint() const override {
        return {ContentWidth() + 2 * Padding(), button_height};
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        const bool hot = isEnabled() && (underMouse() || isDown());
        QColor fill = filled ? Spotlight::Color::Fill() : QColor(Qt::transparent);
        if (hot) {
            fill = filled ? Spotlight::Color::ControlHover() : Spotlight::Color::Fill();
        }
        painter.setPen(hasFocus() ? QPen(Spotlight::Color::Accent(), 1.5) : QPen(Qt::NoPen));
        painter.setBrush(fill);
        const qreal radius = caret ? Spotlight::Radius::Field : height() / 2.0;
        painter.drawRoundedRect(QRectF(rect()).adjusted(1, 1, -1, -1), radius, radius);
        const QColor ink = isEnabled() ? Spotlight::Color::Text() : Spotlight::Color::TextFaint();
        int x = caret ? Padding() : (width() - ContentWidth()) / 2;
        if (!icon.isEmpty()) {
            painter.drawPixmap(x, (height() - 15) / 2,
                               Spotlight::IconPixmap(icon, 15, ink, devicePixelRatioF()));
            x += 15 + (text().isEmpty() ? 0 : 8);
        }
        const int right = caret ? width() - Padding() - 16 - 8 : width();
        painter.setPen(ink);
        painter.drawText(QRect(x, 0, right - x, height()), Qt::AlignVCenter | Qt::AlignLeft,
                         text());
        if (caret) {
            painter.drawPixmap(width() - Padding() - 16, (height() - 16) / 2,
                               Spotlight::IconPixmap(u"caret-down", 16,
                                                     Spotlight::Color::TextSecondary(),
                                                     devicePixelRatioF()));
        }
    }

    void enterEvent(QEnterEvent*) override {
        update();
    }

    void leaveEvent(QEvent*) override {
        update();
    }

private:
    int Padding() const {
        if (caret) {
            return 12;
        }
        return filled ? 18 : 10;
    }

    int ContentWidth() const {
        int content = fontMetrics().horizontalAdvance(text());
        if (!icon.isEmpty()) {
            content += 15 + (text().isEmpty() ? 0 : 8);
        }
        if (caret) {
            content += 16 + 8;
        }
        return content;
    }

    QString icon;
    bool filled;
    bool caret = false;
    int button_height;
};

// One of the two archive formats: an icon tile, the name, a line about it and a radio mark.
class FormatCard final : public QAbstractButton {
public:
    FormatCard(QString icon_, const QString& name, QString about_, bool selected_,
               QWidget* parent)
        : QAbstractButton(parent), icon{std::move(icon_)}, about{std::move(about_)},
          selected{selected_} {
        setText(name);
        setCursor(selected ? Qt::ArrowCursor : Qt::PointingHandCursor);
        setFocusPolicy(selected ? Qt::NoFocus : Qt::StrongFocus);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        setAccessibleName(name);
        setAccessibleDescription(about);
    }

    QSize sizeHint() const override {
        return {280, 82};
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        if (!isEnabled()) {
            painter.setOpacity(0.45);
        }
        const bool hot = !selected && isEnabled() && underMouse();
        const QRectF frame = QRectF(rect()).adjusted(1, 1, -1, -1);
        if (selected) {
            painter.setPen(QPen(Spotlight::Color::Accent(), 2));
            painter.setBrush(QColor(0xff, 0x6b, 0x95, 0x12));
        } else {
            painter.setPen(QPen(hasFocus() ? Spotlight::Color::Accent()
                                           : Spotlight::Color::StrokeSubtle(),
                                1));
            painter.setBrush(hot ? Spotlight::Color::Fill() : Spotlight::Color::FillFaint());
        }
        painter.drawRoundedRect(frame, Spotlight::Radius::Panel, Spotlight::Radius::Panel);

        const qreal ratio = devicePixelRatioF();
        const QRectF tile(14, (height() - 36) / 2.0, 36, 36);
        painter.setPen(Qt::NoPen);
        painter.setBrush(Spotlight::Color::Fill());
        painter.drawRoundedRect(tile, Spotlight::Radius::Field, Spotlight::Radius::Field);
        painter.drawPixmap(tile.topLeft() + QPointF(9, 9),
                           Spotlight::IconPixmap(icon, 18, Spotlight::Color::Text(), ratio));

        const int left = 62;
        const int right = width() - 14 - 22 - 12;
        const QFont name_font = Spotlight::Font(15, 600);
        const QFont about_font = Spotlight::Font(12, 400);
        const QRect about_bounds = QFontMetrics(about_font).boundingRect(
            QRect(left, 0, right - left, 200), Qt::TextWordWrap, about);
        const int text_height = 20 + 2 + about_bounds.height();
        const int top = (height() - text_height) / 2;
        painter.setFont(name_font);
        painter.setPen(Spotlight::Color::Text());
        painter.drawText(QRect(left, top, right - left, 20), Qt::AlignLeft | Qt::AlignVCenter,
                         text());
        painter.setFont(about_font);
        painter.setPen(Spotlight::Color::TextSecondary());
        painter.drawText(QRect(left, top + 22, right - left, about_bounds.height()),
                         Qt::TextWordWrap, about);

        const QRectF mark(width() - 14 - 22, (height() - 22) / 2.0, 22, 22);
        if (selected) {
            painter.setPen(Qt::NoPen);
            painter.setBrush(Spotlight::Color::Accent());
            painter.drawEllipse(mark);
            painter.drawPixmap(mark.topLeft() + QPointF(4.5, 4.5),
                               Spotlight::IconPixmap(u"check", 13, Spotlight::Color::OnAccent(),
                                                     ratio));
        } else {
            painter.setPen(QPen(Spotlight::Color::Stroke(), 2));
            painter.setBrush(Qt::NoBrush);
            painter.drawEllipse(mark.adjusted(1, 1, -1, -1));
        }
    }

    void enterEvent(QEnterEvent*) override {
        update();
    }

    void leaveEvent(QEvent*) override {
        update();
    }

private:
    QString icon;
    QString about;
    bool selected;
};

// A row that opens and closes a panel under it: an icon, a title, a line and a caret.
class DisclosureRow final : public QAbstractButton {
public:
    DisclosureRow(QString icon_, const QString& title, QWidget* parent)
        : QAbstractButton(parent), icon{std::move(icon_)} {
        setText(title);
        setCheckable(true);
        setCursor(Qt::PointingHandCursor);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    }

    void SetDetail(const QString& detail_) {
        detail = detail_;
        update();
    }

    QSize sizeHint() const override {
        return {280, 58};
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(QPen(hasFocus() ? Spotlight::Color::Accent()
                                       : Spotlight::Color::StrokeSubtle(),
                            1));
        painter.setBrush(underMouse() ? Spotlight::Color::FillFaint()
                                      : QColor(0xff, 0xff, 0xff, 0x08));
        painter.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5),
                                Spotlight::Radius::Tile, Spotlight::Radius::Tile);
        const qreal ratio = devicePixelRatioF();
        painter.drawPixmap(14, (height() - 18) / 2,
                           Spotlight::IconPixmap(icon, 18, Spotlight::Color::TextSecondary(),
                                                 ratio));
        const int left = 14 + 18 + 12;
        const int right = width() - 14 - 16 - 12;
        const int top = (height() - 36) / 2;
        painter.setFont(Spotlight::Font(14, 600));
        painter.setPen(Spotlight::Color::Text());
        painter.drawText(QRect(left, top, right - left, 18), Qt::AlignLeft | Qt::AlignVCenter,
                         text());
        painter.setFont(Spotlight::Font(12, 400));
        painter.setPen(Spotlight::Color::TextSecondary());
        painter.drawText(QRect(left, top + 20, right - left, 16), Qt::AlignLeft | Qt::AlignVCenter,
                         detail);
        painter.drawPixmap(width() - 14 - 16, (height() - 16) / 2,
                           Spotlight::IconPixmap(isChecked() ? u"caret-up" : u"caret-down", 16,
                                                 Spotlight::Color::TextSecondary(), ratio));
    }

    void enterEvent(QEnterEvent*) override {
        update();
    }

    void leaveEvent(QEvent*) override {
        update();
    }

private:
    QString icon;
    QString detail;
};

// A path in a rounded field, with an optional button at its end.
class PathRow final : public QWidget {
public:
    PathRow(QStringView icon, QWidget* parent) : QWidget(parent) {
        setFixedHeight(40);
        auto* layout = new QHBoxLayout(this);
        layout->setContentsMargins(14, 0, 6, 0);
        layout->setSpacing(10);
        auto* glyph = new QLabel(this);
        glyph->setPixmap(Spotlight::IconPixmap(icon, 16, Spotlight::Color::TextSecondary(),
                                               devicePixelRatioF()));
        layout->addWidget(glyph);
        path = new ElidedLabel(Spotlight::Font(13, 500), QColor(0xff, 0xff, 0xff, 0xb8), this);
        layout->addWidget(path, 1);
    }

    void SetPath(const QString& text) {
        path->SetText(text);
    }

    void AddButton(QAbstractButton* button) {
        static_cast<QHBoxLayout*>(layout())->addWidget(button);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(QPen(Spotlight::Color::StrokeSubtle(), 1));
        painter.setBrush(QColor(0xff, 0xff, 0xff, 0x08));
        painter.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5),
                                Spotlight::Radius::Field, Spotlight::Radius::Field);
    }

private:
    ElidedLabel* path;
};

// A field with its label above it.
QVBoxLayout* LabeledField(const QString& label, QWidget* field) {
    auto* layout = new QVBoxLayout;
    layout->setSpacing(8);
    layout->addWidget(TextLabel(label, 13, 600, Spotlight::Color::TextSecondary()));
    layout->addWidget(field);
    return layout;
}

// A short centered line under an icon.
QLabel* Caption(QWidget* parent) {
    auto* label = TextLabel({}, 12, 400, Spotlight::Color::TextMuted());
    label->setParent(parent);
    label->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
    label->setWordWrap(true);
    return label;
}

// The format and path at the start of a footer.
QWidget* FileBadge(const QString& format, const QString& path, QWidget* parent) {
    auto* row = new QWidget(parent);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);
    auto* badge = new Box(QColor(0xc9, 0xb3, 0xff, 0x2e), Qt::transparent, 11, row);
    auto* badge_layout = new QHBoxLayout(badge);
    badge_layout->setContentsMargins(9, 0, 9, 0);
    badge_layout->addWidget(TextLabel(format, 11, 600, QColor(0xc9, 0xb3, 0xff)));
    badge->setFixedHeight(22);
    layout->addWidget(badge);
    auto* text = new ElidedLabel(Spotlight::Font(12, 400), Spotlight::Color::TextMuted(), row);
    text->SetText(path);
    layout->addWidget(text, 1);
    return row;
}

// A green strip with a check and a line, such as what saving does.
QWidget* NoteStrip(const QString& text, QWidget* parent) {
    auto* strip = new Box(QColor(0x7e, 0xe2, 0xa8, 0x14), QColor(0x7e, 0xe2, 0xa8, 0x33),
                          Spotlight::Radius::Tile, parent);
    auto* layout = new QHBoxLayout(strip);
    layout->setContentsMargins(14, 12, 14, 12);
    layout->setSpacing(10);
    auto* glyph = new QLabel(strip);
    glyph->setPixmap(Spotlight::IconPixmap(u"check-circle", 16, Spotlight::Color::Success(),
                                           strip->devicePixelRatioF()));
    layout->addWidget(glyph, 0, Qt::AlignTop);
    auto* label = TextLabel(text, 13, 400, QColor(0xff, 0xff, 0xff, 0xb8));
    label->setWordWrap(true);
    layout->addWidget(label, 1);
    return strip;
}

// Shows a picture with a square selection: click or drag to move it.
class CropArea final : public QWidget {
public:
    CropArea(const QImage& image_, QWidget* parent) : QWidget(parent), image{image_} {
        const QSize fit = image.size().scaled(472, 300, Qt::KeepAspectRatio);
        setFixedSize(fit);
        scale = static_cast<double>(fit.width()) / image.width();
        side = std::min(image.width(), image.height());
        center = QPointF(image.width() / 2.0, image.height() / 2.0);
        setCursor(Qt::SizeAllCursor);
        setFocusPolicy(Qt::StrongFocus);
    }

    // 100: the whole short side; 400: a quarter of it.
    void SetZoom(int percent) {
        side = std::min(image.width(), image.height()) * 100.0 / percent;
        Clamp();
    }

    QRect Selection() const {
        const int s = static_cast<int>(side);
        const int x = std::clamp(static_cast<int>(center.x() - side / 2.0), 0, image.width() - s);
        const int y = std::clamp(static_cast<int>(center.y() - side / 2.0), 0, image.height() - s);
        return QRect(x, y, s, s);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setRenderHint(QPainter::SmoothPixmapTransform);
        QPainterPath frame;
        frame.addRoundedRect(QRectF(rect()), Spotlight::Radius::Panel, Spotlight::Radius::Panel);
        painter.setClipPath(frame);
        painter.drawImage(rect(), image);
        const QRect s = Selection();
        const QRectF shown(s.x() * scale, s.y() * scale, s.width() * scale, s.height() * scale);
        const qreal radius = shown.width() * 28.0 / 240.0;
        QPainterPath outside;
        outside.addRect(QRectF(rect()));
        outside.addRoundedRect(shown, radius, radius);
        painter.fillPath(outside, QColor(0, 0, 0, 0x99));
        painter.setPen(QPen(hasFocus() ? Spotlight::Color::Accent() : Spotlight::Color::Text(), 2));
        painter.setBrush(Qt::NoBrush);
        painter.drawRoundedRect(shown.adjusted(1, 1, -1, -1), radius, radius);
    }
    void mousePressEvent(QMouseEvent* event) override {
        last = event->position();
        pressed = event->position();
    }
    void mouseMoveEvent(QMouseEvent* event) override {
        if ((event->buttons() & Qt::LeftButton) == 0) {
            return;
        }
        center += (event->position() - last) / scale;
        last = event->position();
        Clamp();
    }
    void mouseReleaseEvent(QMouseEvent* event) override {
        // A simple click centers the square on the point.
        if (event->position() == pressed) {
            center = event->position() / scale;
            Clamp();
        }
    }
    void keyPressEvent(QKeyEvent* event) override {
        const double step = side / 20.0;
        switch (event->key()) {
        case Qt::Key_Left:
            center.rx() -= step;
            break;
        case Qt::Key_Right:
            center.rx() += step;
            break;
        case Qt::Key_Up:
            center.ry() -= step;
            break;
        case Qt::Key_Down:
            center.ry() += step;
            break;
        default:
            QWidget::keyPressEvent(event);
            return;
        }
        Clamp();
    }

private:
    void Clamp() {
        center.setX(std::clamp(center.x(), side / 2.0, image.width() - side / 2.0));
        center.setY(std::clamp(center.y(), side / 2.0, image.height() - side / 2.0));
        update();
    }

    QImage image;
    double scale = 1.0;
    double side = 0.0;
    QPointF center;
    QPointF last;
    QPointF pressed;
};

class CropDialog final : public ArchiveCardDialog {
public:
    CropDialog(const QImage& image, QWidget* parent)
        : ArchiveCardDialog(tr("Crop icon"), 520, parent) {
        SetHeader(QStringLiteral("crop"),
                  tr("Drag the square to pick the part that becomes the icon."));
        area = new CropArea(image, card);
        body->addWidget(area, 0, Qt::AlignHCenter);

        auto* zoom_row = new QHBoxLayout;
        zoom_row->setSpacing(12);
        auto* zoom = new QSlider(Qt::Horizontal, card);
        zoom->setRange(100, 400);
        zoom->setSingleStep(10);
        zoom->setPageStep(20);
        zoom->setAccessibleName(tr("Zoom"));
        auto* out = new FlatButton({}, QStringLiteral("magnifying-glass-minus"), false, 28, card);
        auto* in = new FlatButton({}, QStringLiteral("magnifying-glass-plus"), false, 28, card);
        out->setToolTip(tr("Zoom out"));
        in->setToolTip(tr("Zoom in"));
        auto* percent = TextLabel(QStringLiteral("100%"), 13, 600, Spotlight::Color::Text());
        percent->setFixedWidth(44);
        percent->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        connect(zoom, &QSlider::valueChanged, area, [this, percent](int value) {
            area->SetZoom(value);
            percent->setText(QStringLiteral("%1%").arg(value));
        });
        connect(out, &QAbstractButton::clicked, zoom,
                [zoom] { zoom->setValue(zoom->value() - zoom->pageStep()); });
        connect(in, &QAbstractButton::clicked, zoom,
                [zoom] { zoom->setValue(zoom->value() + zoom->pageStep()); });
        zoom_row->addWidget(out);
        zoom_row->addWidget(zoom, 1);
        zoom_row->addWidget(in);
        zoom_row->addWidget(percent);
        body->addLayout(zoom_row);
        AddButtons(tr("Use this crop"), {});
        area->setFocus();
    }

    QRect Selection() const {
        return area->Selection();
    }

private:
    CropArea* area;
};

} // namespace

std::optional<QRect> ChooseIconCrop(const QImage& image, QWidget* parent) {
    if (image.width() == image.height()) {
        return image.rect();
    }
    CropDialog dialog(image, parent);
    if (dialog.exec() != QDialog::Accepted) {
        return std::nullopt;
    }
    return dialog.Selection();
}

QByteArray EncodeArchiveIcon(const QImage& source, const QRect& crop, int size) {
    const QImage scaled =
        source.copy(crop).scaled(size, size, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QImage flat(size, size, QImage::Format_RGB32);
    flat.fill(Qt::black);
    QPainter painter(&flat);
    painter.drawImage(0, 0, scaled);
    painter.end();
    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    if (!flat.save(&buffer, "JPEG", 92)) {
        return {};
    }
    return bytes;
}

IconDropZone::IconDropZone(int size, QWidget* parent) : QWidget(parent) {
    setFixedSize(size, size);
    setAcceptDrops(true);
    setCursor(Qt::PointingHandCursor);
    setToolTip(tr("Drop an image here or click to choose one"));
}

void IconDropZone::SetIcon(const QPixmap& new_icon) {
    icon = new_icon;
    update();
}

void IconDropZone::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    const QRectF area = QRectF(rect()).adjusted(1, 1, -1, -1);
    const qreal radius = width() >= 120 ? 24 : Spotlight::Radius::Tile;
    const bool active = dragging || hovered;
    if (!icon.isNull()) {
        painter.drawPixmap(0, 0, RoundedPreview(icon, width(), radius));
        painter.setPen(QPen(Spotlight::Color::StrokeSubtle(), 1));
        painter.setBrush(Qt::NoBrush);
        painter.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5), radius, radius);
        if (active) {
            painter.setBrush(QColor(0, 0, 0, 130));
            painter.setPen(Qt::NoPen);
            painter.drawRoundedRect(area, radius, radius);
            painter.setPen(Spotlight::Color::Text());
            painter.setFont(Spotlight::Font(12, 600));
            painter.drawText(area, Qt::AlignCenter | Qt::TextWordWrap,
                             dragging ? tr("Drop to use") : tr("Change"));
        }
        return;
    }
    const auto accent = Spotlight::Color::Accent();
    QPen pen(active ? accent : Spotlight::Color::Stroke(), 1.5, Qt::DashLine);
    painter.setPen(pen);
    painter.setBrush(active ? Spotlight::Color::AccentSoft() : Spotlight::Color::FillFaint());
    painter.drawRoundedRect(area, radius, radius);
    const QPixmap glyph = Spotlight::IconPixmap(u"image", 24, Spotlight::Color::TextMuted(),
                                                devicePixelRatioF());
    painter.drawPixmap(QPointF((width() - 24) / 2.0, height() / 2.0 - 26), glyph);
    painter.setPen(Spotlight::Color::TextMuted());
    painter.setFont(Spotlight::Font(11, 500));
    painter.drawText(area.adjusted(6, height() / 2.0, -6, -6), Qt::AlignHCenter | Qt::AlignTop |
                                                                   Qt::TextWordWrap,
                     dragging ? tr("Release to use") : tr("Add an icon"));
}

void IconDropZone::dragEnterEvent(QDragEnterEvent* event) {
    if (!DroppedImage(event->mimeData()).isEmpty()) {
        dragging = true;
        event->acceptProposedAction();
        update();
    }
}

void IconDropZone::dragLeaveEvent(QDragLeaveEvent*) {
    dragging = false;
    update();
}

void IconDropZone::dropEvent(QDropEvent* event) {
    dragging = false;
    update();
    if (const auto path = DroppedImage(event->mimeData()); !path.isEmpty()) {
        event->acceptProposedAction();
        emit FileDropped(path);
    }
}

void IconDropZone::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && rect().contains(event->position().toPoint())) {
        emit Clicked();
    }
}

void IconDropZone::enterEvent(QEnterEvent*) {
    hovered = true;
    update();
}

void IconDropZone::leaveEvent(QEvent*) {
    hovered = false;
    update();
}

ArchiveCardDialog::ArchiveCardDialog(const QString& title, int width, QWidget* parent)
    : Spotlight::Dialog(parent) {
    setStyleSheet(QString::fromLatin1(CardStyle));
    setWindowTitle(title);
    card = new QWidget(this);
    card->setObjectName(QStringLiteral("archiveCard"));
    card->setFixedWidth(width - 2 * CardPadding);
    body = new QVBoxLayout(card);
    body->setContentsMargins(0, 0, 0, 0);
    body->setSpacing(14);
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->addWidget(card);
}

void ArchiveCardDialog::SetHeader(const QString& icon, const QString& subtitle) {
    SetIcon(icon, Spotlight::Color::Accent());
    SetSubtitle(subtitle);
}

QVBoxLayout* ArchiveCardDialog::AddSection(const QString& title) {
    auto* group = new QVBoxLayout;
    group->setSpacing(10);
    if (!title.isEmpty()) {
        group->addWidget(TextLabel(title, 13, 600, Spotlight::Color::TextSecondary()));
    }
    auto* section = new Box(Spotlight::Color::FillFaint(), Spotlight::Color::StrokeSubtle(),
                            Spotlight::Radius::Panel, card);
    auto* layout = new QVBoxLayout(section);
    layout->setContentsMargins(14, 12, 14, 12);
    layout->setSpacing(10);
    group->addWidget(section);
    body->addLayout(group);
    return layout;
}

QLabel* ArchiveCardDialog::Hint(const QString& text) {
    auto* label = TextLabel(text, 12, 400, Spotlight::Color::TextSecondary());
    label->setWordWrap(true);
    return label;
}

Spotlight::PillButton* ArchiveCardDialog::AddButtons(const QString& action,
                                                     const QString& action_icon,
                                                     QWidget* leading) {
    auto* row = new QHBoxLayout;
    row->setSpacing(10);
    if (leading != nullptr) {
        row->addWidget(leading, 1);
    } else {
        row->addStretch();
    }
    auto* cancel_button = new Spotlight::PillButton(tr("Cancel"), {},
                                                    Spotlight::PillButton::Kind::Control, 40, card);
    action_button = new Spotlight::PillButton(action, action_icon,
                                              Spotlight::PillButton::Kind::Accent, 40, card);
    row->addWidget(cancel_button);
    row->addWidget(action_button);
    body->addSpacing(6);
    body->addLayout(row);
    connect(cancel_button, &QAbstractButton::clicked, this, &QDialog::reject);
    connect(action_button, &QAbstractButton::clicked, this, &QDialog::accept);
    return action_button;
}

void ArchiveCardDialog::keyPressEvent(QKeyEvent* event) {
    if ((event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) &&
        action_button != nullptr) {
        if (action_button->isEnabled()) {
            action_button->click();
        }
        return;
    }
    QDialog::keyPressEvent(event);
}

ArchiveMessageDialog::ArchiveMessageDialog(const QString& title, const QString& text,
                                           QWidget* parent, ArchiveTone tone)
    : ArchiveCardDialog(title, 460, parent) {
    SetSubtitle(text);
    switch (tone) {
    case ArchiveTone::None:
        break;
    case ArchiveTone::Success:
        SetIcon(u"check-circle", Spotlight::Color::Success());
        break;
    case ArchiveTone::Error:
        SetIcon(u"warning-circle", Spotlight::Color::AccentLight());
        break;
    case ArchiveTone::Warning:
        SetIcon(u"warning-circle", Spotlight::Color::Warning());
        break;
    }
}

QCheckBox* ArchiveMessageDialog::AddCheckBox(const QString& text, bool checked) {
    if (checks == nullptr) {
        checks = AddSection({});
    }
    auto* box = new QCheckBox(text, card);
    box->setChecked(checked);
    checks->addWidget(box);
    return box;
}

QAbstractButton* ArchiveMessageDialog::AddButton(const QString& text, const QString& icon,
                                                 bool accent) {
    if (buttons == nullptr) {
        buttons = new QHBoxLayout;
        buttons->setSpacing(10);
        buttons->addStretch();
        body->addSpacing(4);
        body->addLayout(buttons);
    }
    auto* button = new Spotlight::PillButton(text, icon,
                                             accent ? Spotlight::PillButton::Kind::Accent
                                                    : Spotlight::PillButton::Kind::Control,
                                             40, card);
    buttons->addWidget(button);
    if (accent) {
        action_button = button;
    }
    connect(button, &QAbstractButton::clicked, this, [this, button, accent] {
        clicked = button;
        accent ? accept() : reject();
    });
    return button;
}

void ArchiveNotice(QWidget* parent, const QString& title, const QString& text, ArchiveTone tone) {
    ArchiveMessageDialog dialog(title, text, parent, tone);
    dialog.AddButton(tone == ArchiveTone::Error ? QObject::tr("Close") : QObject::tr("Done"), {},
                     true);
    dialog.exec();
}

ArchiveConvertDialog::ArchiveConvertDialog(Format format_, bool offer_other, QWidget* parent)
    : ArchiveCardDialog(tr("Convert to a single file"), 680, parent), format{format_} {
    SetHeader(QStringLiteral("archive"),
              tr("Packs this game into one archive. The game itself is not changed."));
    const bool nup = format == Format::Nup;
    auto* formats = new QHBoxLayout;
    formats->setSpacing(12);
    auto* nup_card = new FormatCard(
        QStringLiteral("archive"), tr("NUP"),
        tr("Lichii archive for loose dumps. Fast to load, keeps updates."), nup, card);
    auto* nx_card = new FormatCard(QStringLiteral("cube"), tr("NX"),
                                   tr("Compressed archive from a dump, NUP, NSP or XCI."), !nup,
                                   card);
    auto* other = nup ? nx_card : nup_card;
    other->setEnabled(offer_other);
    connect(other, &QAbstractButton::clicked, this, [this] { done(OtherFormat); });
    formats->addWidget(nup_card);
    formats->addWidget(nx_card);
    body->addLayout(formats);
}

void ArchiveConvertDialog::AddFiles(const QString& packages_title,
                                    std::vector<ArchivePackage> packages_, const QString& source,
                                    std::filesystem::path archive_, bool offer_delete,
                                    u64 extra_bytes_, const QString& action) {
    packages = std::move(packages_);
    archive = std::move(archive_);
    extra_bytes = extra_bytes_;
    package_checks.assign(packages.size(), nullptr);
    if (std::ranges::any_of(packages, &ArchivePackage::optional)) {
        auto* group = new QVBoxLayout;
        group->setSpacing(8);
        group->addWidget(TextLabel(packages_title, 13, 600, Spotlight::Color::TextSecondary()));
        auto* list = new Box(Spotlight::Color::FillFaint(), Spotlight::Color::StrokeSubtle(),
                             Spotlight::Radius::Panel, card);
        auto* frame = new QVBoxLayout(list);
        frame->setContentsMargins(0, 3, 0, 3);
        auto* scroll = new QScrollArea(list);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto* inner = new QWidget;
        auto* rows = new QVBoxLayout(inner);
        rows->setContentsMargins(14, 0, 14, 0);
        rows->setSpacing(0);
        int shown = 0;
        for (size_t i = 0; i < packages.size(); ++i) {
            const auto& package = packages[i];
            if (!package.optional) {
                continue;
            }
            ++shown;
            auto* row = new QHBoxLayout;
            row->setContentsMargins(0, 5, 0, 5);
            row->setSpacing(10);
            auto* check = new QCheckBox(package.label, inner);
            check->setFont(Spotlight::Font(14, 600));
            check->setChecked(true);
            connect(check, &QAbstractButton::toggled, this, &ArchiveConvertDialog::UpdateSpace);
            row->addWidget(check);
            auto* detail = new ElidedLabel(Spotlight::Font(12, 400), Spotlight::Color::TextMuted(),
                                           inner);
            detail->SetText(package.detail);
            row->addWidget(detail, 1);
            row->addWidget(TextLabel(HumanSize(package.bytes), 12, 500,
                                     Spotlight::Color::TextSecondary()));
            rows->addLayout(row);
            package_checks[i] = check;
        }
        scroll->setWidget(inner);
        // Two rows show and the rest scroll, so the card fits an 800 px window.
        scroll->setFixedHeight(inner->sizeHint().height() * std::min(shown, 2) / shown);
        frame->addWidget(scroll);
        group->addWidget(list);
        body->addLayout(group);
    }

    auto* files = new QVBoxLayout;
    files->setSpacing(8);
    files->addWidget(TextLabel(tr("Files"), 13, 600, Spotlight::Color::TextSecondary()));
    const auto add_file_row = [&](const QString& label, QWidget* field) {
        auto* row = new QHBoxLayout;
        row->setSpacing(10);
        auto* text = TextLabel(label, 12, 600, Spotlight::Color::TextMuted());
        text->setFixedWidth(40);
        row->addWidget(text);
        row->addWidget(field, 1);
        files->addLayout(row);
    };
    auto* from = new PathRow(u"folder", card);
    from->SetPath(source);
    auto* to = new PathRow(u"file", card);
    auto* change = new FlatButton(tr("Change…"), {}, true, 28, card);
    connect(change, &QAbstractButton::clicked, this, &ArchiveConvertDialog::ChooseTarget);
    to->AddButton(change);
    target = to;
    add_file_row(tr("From"), from);
    add_file_row(tr("To"), to);
    space_label = Hint({});
    if (offer_delete) {
        auto* space_row = new QHBoxLayout;
        space_row->setSpacing(10);
        space_row->addWidget(space_label, 1);
        delete_sources = new QCheckBox(tr("Delete the original files while packing"), card);
        delete_sources->setToolTip(
            tr("Deletes each loose file once it is in the archive. This cannot be undone."));
        connect(delete_sources, &QAbstractButton::toggled, this,
                &ArchiveConvertDialog::UpdateSpace);
        space_row->addWidget(delete_sources);
        files->addLayout(space_row);
    }
    body->addLayout(files);
    AddButtons(action, QStringLiteral("archive"), offer_delete ? nullptr : space_label);
    UpdateSpace();
}

std::filesystem::path ArchiveConvertDialog::Archive() const {
    return archive;
}

bool ArchiveConvertDialog::DeleteSources() const {
    return delete_sources != nullptr && delete_sources->isChecked();
}

std::vector<bool> ArchiveConvertDialog::IncludedPackages() const {
    std::vector<bool> included;
    for (const auto* check : package_checks) {
        included.push_back(check == nullptr || check->isChecked());
    }
    return included;
}

void ArchiveConvertDialog::ChooseTarget() {
    const bool nup = format == Format::Nup;
    const QString extension = nup ? QStringLiteral(".nup") : QStringLiteral(".nx");
    QString chosen = QFileDialog::getSaveFileName(
        this, nup ? tr("Save the NUP as") : tr("Save the NX archive as"), ToQString(archive),
        nup ? tr("NUP archive (*.nup)") : tr("NX archive (*.nx)"));
    if (chosen.isEmpty()) {
        return;
    }
    if (!chosen.endsWith(extension, Qt::CaseInsensitive)) {
        chosen += extension;
    }
    archive = FromQString(chosen);
    UpdateSpace();
}

void ArchiveConvertDialog::UpdateSpace() {
    static_cast<PathRow*>(target)->SetPath(ToQString(archive));
    u64 needed = extra_bytes;
    if (!DeleteSources()) {
        const auto included = IncludedPackages();
        for (size_t i = 0; i < packages.size(); ++i) {
            if (included[i]) {
                needed += packages[i].bytes;
            }
        }
    }
    const u64 available = Common::FS::GetFreeSpaceSize(archive.parent_path());
    std::error_code ec;
    QString problem;
    if (std::filesystem::exists(archive, ec)) {
        problem = tr("A file with this name already exists.");
    } else if (format == Format::Nup && available < needed) {
        problem = tr("Not enough free space on this drive.");
    }
    QString text =
        tr("About %1 · %2 free on this drive").arg(HumanSize(needed), HumanSize(available));
    if (!problem.isEmpty()) {
        text = problem + QLatin1Char(' ') + text;
    }
    space_label->setText(text);
    Spotlight::SetTextColor(space_label, problem.isEmpty() ? Spotlight::Color::TextSecondary()
                                                           : Spotlight::Color::Warning());
    if (action_button != nullptr) {
        action_button->setEnabled(problem.isEmpty());
    }
}

NupConvertDialog::NupConvertDialog(NupConvertSummary summary_, QWidget* parent)
    : ArchiveConvertDialog(Format::Nup, summary_.offer_nx, parent), summary{std::move(summary_)} {
    const auto& detected = summary.detected;
    auto* details = new QHBoxLayout;
    details->setSpacing(22);

    auto* column = new QVBoxLayout;
    column->setSpacing(12);
    icon_preview = new IconDropZone(ColumnWidth, card);
    connect(icon_preview, &IconDropZone::Clicked, this, &NupConvertDialog::ChooseIcon);
    connect(icon_preview, &IconDropZone::FileDropped, this, &NupConvertDialog::SetIconFromFile);
    recrop = new Spotlight::CircleButton(QStringLiteral("crop"), 34, icon_preview);
    recrop->setToolTip(tr("Crop again"));
    recrop->move(ColumnWidth - 42, ColumnWidth - 42);
    connect(recrop, &QAbstractButton::clicked, this, &NupConvertDialog::Recrop);
    icon_caption = Caption(card);
    icon_caption->setFixedWidth(ColumnWidth);
    auto* choose = new FlatButton(tr("Choose image…"), QStringLiteral("image"), true, 34, card);
    choose->setFixedWidth(ColumnWidth);
    connect(choose, &QAbstractButton::clicked, this, &NupConvertDialog::ChooseIcon);
    reset_icon = new FlatButton(tr("Use the game's icon"),
                                QStringLiteral("arrow-counter-clockwise"), false, 30, card);
    connect(reset_icon, &QAbstractButton::clicked, this, &NupConvertDialog::ResetIcon);
    column->addWidget(icon_preview);
    column->addWidget(icon_caption);
    column->addWidget(choose);
    column->addWidget(reset_icon, 0, Qt::AlignHCenter);
    column->addStretch();
    details->addLayout(column);

    auto* fields = new QVBoxLayout;
    fields->setSpacing(12);
    name_edit = new QLineEdit(QString::fromStdString(detected.title_name), card);
    name_edit->setPlaceholderText(tr("Game name"));
    name_edit->setClearButtonEnabled(true);
    publisher_edit = new QLineEdit(QString::fromStdString(detected.publisher), card);
    publisher_edit->setPlaceholderText(tr("Publisher"));
    publisher_edit->setClearButtonEnabled(true);
    fields->addLayout(LabeledField(tr("Name"), name_edit));
    fields->addLayout(LabeledField(tr("Publisher"), publisher_edit));

    auto* version = new QVBoxLayout;
    version->setSpacing(10);
    auto_version = new QRadioButton(
        detected.display_version.empty()
            ? tr("Auto-detect: not found")
            : tr("Auto-detect: %1").arg(QString::fromStdString(detected.display_version)),
        card);
    auto_version->setFont(Spotlight::Font(14, 600));
    auto* auto_row = new QHBoxLayout;
    auto_row->setSpacing(8);
    auto_row->addWidget(auto_version);
    if (!detected.version_source.empty()) {
        auto* source = new ElidedLabel(Spotlight::Font(13, 400), Spotlight::Color::TextSecondary(),
                                       card);
        source->SetText(tr("from the %1").arg(QString::fromStdString(detected.version_source)));
        auto_row->addWidget(source, 1);
    } else {
        auto_row->addStretch();
    }
    version->addLayout(auto_row);
    auto* manual_row = new QHBoxLayout;
    manual_row->setSpacing(12);
    manual_version = new QRadioButton(tr("Set manually"), card);
    manual_version->setFont(Spotlight::Font(14, 600));
    version_edit = new QLineEdit(card);
    version_edit->setPlaceholderText(tr("e.g. 1.2.0"));
    version_edit->setFixedWidth(160);
    manual_row->addWidget(manual_version);
    manual_row->addWidget(version_edit);
    manual_row->addStretch();
    version->addLayout(manual_row);
    auto* version_group = new QButtonGroup(this);
    version_group->addButton(auto_version);
    version_group->addButton(manual_version);
    (detected.display_version.empty() ? manual_version : auto_version)->setChecked(true);
    connect(version_edit, &QLineEdit::textEdited, manual_version,
            [this] { manual_version->setChecked(true); });
    auto* version_field = new QVBoxLayout;
    version_field->setSpacing(8);
    version_field->addWidget(TextLabel(tr("Version"), 13, 600, Spotlight::Color::TextSecondary()));
    version_field->addLayout(version);
    fields->addLayout(version_field);
    fields->addStretch();
    details->addLayout(fields, 1);
    body->addLayout(details);

    AddFiles(tr("Packed into the NUP"), summary.packages, summary.source_name, summary.archive,
             summary.offer_low_storage, summary.extra_bytes, tr("Convert to NUP"));
    UpdateIconPreview();
    name_edit->setFocus();
}

NupConvertDialog::~NupConvertDialog() = default;

FileSys::Nup::Metadata NupConvertDialog::Metadata() const {
    const auto& detected = summary.detected;
    FileSys::Nup::Metadata result;
    if (const auto name = name_edit->text().trimmed().toStdString(); name != detected.title_name) {
        result.title_name = name;
    }
    if (const auto publisher = publisher_edit->text().trimmed().toStdString();
        publisher != detected.publisher) {
        result.publisher = publisher;
    }
    if (manual_version->isChecked()) {
        if (const auto version = version_edit->text().trimmed().toStdString();
            version != detected.display_version) {
            result.display_version = version;
        }
    }
    if (!custom_icon.isEmpty()) {
        result.icon.assign(custom_icon.begin(), custom_icon.end());
    }
    return result;
}

QString NupConvertDialog::DisplayName() const {
    return name_edit->text().trimmed();
}

QPixmap NupConvertDialog::CurrentIcon() const {
    return current_icon;
}

void NupConvertDialog::ChooseIcon() {
    const auto path = QFileDialog::getOpenFileName(
        this, tr("Choose Icon"), {}, tr("Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"));
    if (!path.isEmpty()) {
        SetIconFromFile(path);
    }
}

void NupConvertDialog::SetIconFromFile(const QString& path) {
    const QImage image = ReadImage(path);
    if (image.isNull()) {
        icon_caption->setText(tr("That image could not be read."));
        return;
    }
    const auto crop = ChooseIconCrop(image, this);
    if (!crop) {
        return;
    }
    icon_source = image;
    icon_crop = *crop;
    custom_icon = EncodeArchiveIcon(image, *crop);
    UpdateIconPreview();
}

void NupConvertDialog::Recrop() {
    if (const auto crop = ChooseIconCrop(icon_source, this)) {
        icon_crop = *crop;
        custom_icon = EncodeArchiveIcon(icon_source, icon_crop);
        UpdateIconPreview();
    }
}

void NupConvertDialog::ResetIcon() {
    icon_source = {};
    custom_icon.clear();
    UpdateIconPreview();
}

void NupConvertDialog::UpdateIconPreview() {
    QPixmap pixmap;
    if (!custom_icon.isEmpty()) {
        pixmap.loadFromData(custom_icon);
        icon_caption->setText(tr("Custom image · %1 × %1").arg(pixmap.width()));
    } else {
        pixmap = PixmapFromBytes(summary.detected.icon);
        icon_caption->setText(pixmap.isNull()
                                  ? tr("No icon found. Drop an image on the square.")
                                  : tr("Detected from the game · %1 × %1").arg(pixmap.width()));
    }
    current_icon = pixmap;
    reset_icon->setVisible(!custom_icon.isEmpty());
    reset_icon->setText(summary.detected.icon.empty() ? tr("Remove") : tr("Use the game's icon"));
    recrop->setVisible(!icon_source.isNull() && icon_source.width() != icon_source.height());
    icon_preview->SetIcon(pixmap);
}

NxConvertDialog::NxConvertDialog(const QString& source, std::vector<ArchivePackage> related,
                                 std::filesystem::path archive, bool offer_nup, QWidget* parent)
    : ArchiveConvertDialog(Format::Nx, offer_nup, parent) {
    AddFiles(tr("Also pack these, found in the same folder"), std::move(related), source,
             std::move(archive), false, 0, tr("Convert to NX"));
}

ArchiveEditDialog::ArchiveEditDialog(Format format_, const std::filesystem::path& archive,
                                     FileSys::Nup::Metadata current_,
                                     std::vector<u8> original_icon_, QWidget* parent)
    : ArchiveCardDialog(tr("Title and icon"), 700, parent), format{format_},
      current{std::move(current_)}, original_icon{std::move(original_icon_)} {
    const bool nup = format == Format::Nup;
    SetHeader(QStringLiteral("pencil-simple"),
              tr("Saved into %1, so every app that reads the file shows them.")
                  .arg(ToQString(archive.filename())));
    auto* details = new QHBoxLayout;
    details->setSpacing(22);

    auto* column = new QVBoxLayout;
    column->setSpacing(12);
    icon_preview = new IconDropZone(ColumnWidth, card);
    connect(icon_preview, &IconDropZone::Clicked, this, &ArchiveEditDialog::ChooseIcon);
    connect(icon_preview, &IconDropZone::FileDropped, this, &ArchiveEditDialog::SetIconFromFile);
    recrop = new Spotlight::CircleButton(QStringLiteral("crop"), 34, icon_preview);
    recrop->setToolTip(tr("Crop again"));
    recrop->move(ColumnWidth - 42, ColumnWidth - 42);
    connect(recrop, &QAbstractButton::clicked, this, &ArchiveEditDialog::Recrop);
    icon_hint = Caption(card);
    icon_hint->setFixedWidth(ColumnWidth);
    auto* choose = new FlatButton(tr("Choose image…"), QStringLiteral("image"), true, 34, card);
    choose->setFixedWidth(ColumnWidth);
    connect(choose, &QAbstractButton::clicked, this, &ArchiveEditDialog::ChooseIcon);
    find_icon =
        new FlatButton(tr("Find online…"), QStringLiteral("magnifying-glass"), true, 34, card);
    find_icon->setFixedWidth(ColumnWidth);
    connect(find_icon, &QAbstractButton::clicked, this, &ArchiveEditDialog::FindIcon);
    restore_icon = new FlatButton(tr("Restore"), QStringLiteral("arrow-counter-clockwise"), false,
                                  30, card);
    connect(restore_icon, &QAbstractButton::clicked, this, &ArchiveEditDialog::RestoreIcon);
    clear_icon = new FlatButton(tr("Remove"), QStringLiteral("trash"), false, 30, card);
    connect(clear_icon, &QAbstractButton::clicked, this, &ArchiveEditDialog::ClearIcon);
    auto* small = new QHBoxLayout;
    small->setSpacing(4);
    small->addWidget(restore_icon);
    small->addWidget(clear_icon);
    small->addStretch();
    column->addWidget(icon_preview);
    column->addWidget(icon_hint);
    column->addWidget(choose);
    column->addWidget(find_icon);
    column->addLayout(small);
    column->addStretch();
    details->addLayout(column);

    icon_search = new IconSearch(this);
    connect(icon_search, &IconSearch::RunningChanged, this, [this](bool running) {
        find_icon->setEnabled(!running);
        find_icon->setText(running ? tr("Searching…") : tr("Find online…"));
    });
    connect(icon_search, &IconSearch::IconSelected, this, &ArchiveEditDialog::OnIconFound);

    auto* fields = new QVBoxLayout;
    fields->setSpacing(14);
    name_edit = new QLineEdit(QString::fromStdString(current.title_name), card);
    name_edit->setObjectName(QStringLiteral("nupEditName"));
    name_edit->setPlaceholderText(tr("Game name"));
    name_edit->setClearButtonEnabled(true);
    fields->addLayout(LabeledField(tr("Name"), name_edit));
    publisher_edit = new QLineEdit(QString::fromStdString(current.publisher), card);
    publisher_edit->setObjectName(QStringLiteral("nupEditPublisher"));
    publisher_edit->setPlaceholderText(tr("Publisher"));
    publisher_edit->setClearButtonEnabled(true);
    fields->addLayout(LabeledField(tr("Publisher"), publisher_edit));

    auto* pair = new QHBoxLayout;
    pair->setSpacing(14);
    if (nup) {
        version_edit = new QLineEdit(QString::fromStdString(current.display_version), card);
        version_edit->setPlaceholderText(tr("e.g. 1.2.0"));
        version_edit->setFixedWidth(150);
        pair->addLayout(LabeledField(tr("Version"), version_edit));
    } else {
        auto* size_button = new FlatButton(tr("256 × 256"), {}, true, 34, card);
        size_button->SetCaret(true);
        size_button->setFixedWidth(160);
        size_button->setToolTip(tr("Pixels per side of the icon stored in the archive"));
        auto* menu = new Spotlight::Menu(size_button);
        for (int index = 0; index < 3; ++index) {
            const int side = 256 << index;
            menu->addAction(tr("%1 × %1").arg(side), this, [this, size_button, index, side] {
                icon_size_index = index;
                size_button->setText(tr("%1 × %1").arg(side));
                EncodeIcon();
            });
        }
        connect(size_button, &QAbstractButton::clicked, menu, [size_button, menu] {
            menu->popup(size_button->mapToGlobal(QPoint(0, size_button->height() + 6)));
        });
        pair->addLayout(LabeledField(tr("Icon size"), size_button));
    }
    pair->addStretch();
    fields->addLayout(pair);

    if (nup) {
        auto* languages = new DisclosureRow(QStringLiteral("translate"),
                                            tr("Names in other languages"), card);
        fields->addWidget(languages);
        auto* scroll = new QScrollArea(card);
        localized_panel = scroll;
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setFixedHeight(200);
        auto* inner = new QWidget;
        auto* inner_layout = new QVBoxLayout(inner);
        inner_layout->setContentsMargins(0, 0, 8, 0);
        inner_layout->setSpacing(8);
        for (const auto& language : LocalizedLanguages) {
            auto* field = new QLineEdit(inner);
            field->setClearButtonEnabled(true);
            for (const auto& [tag, name] : current.localized_titles) {
                if (tag == language.tag) {
                    field->setText(QString::fromStdString(name));
                }
            }
            auto* row = new QHBoxLayout;
            auto* label = TextLabel(tr(language.name), 12, 500, Spotlight::Color::TextMuted());
            label->setFixedWidth(150);
            row->addWidget(label);
            row->addWidget(field, 1);
            inner_layout->addLayout(row);
            localized_edits.emplace_back(language.tag, field);
        }
        scroll->setWidget(inner);
        fields->addWidget(localized_panel);
        localized_panel->setVisible(false);
        connect(languages, &QAbstractButton::toggled, this, [this](bool shown) {
            localized_panel->setVisible(shown);
            adjustSize();
        });
        const auto update_languages = [this, languages] {
            const QString fallback = name_edit->text().trimmed();
            int set = 0;
            for (const auto& [tag, field] : localized_edits) {
                field->setPlaceholderText(fallback.isEmpty() ? tr("Same as Name") : fallback);
                set += field->text().trimmed().isEmpty() ? 0 : 1;
            }
            languages->SetDetail(set == 0 ? tr("None set · every language uses the name above")
                                          : tr("%1 set · others use the name above").arg(set));
        };
        connect(name_edit, &QLineEdit::textChanged, this, update_languages);
        for (const auto& [tag, field] : localized_edits) {
            connect(field, &QLineEdit::textChanged, this, update_languages);
        }
        update_languages();
    }
    fields->addStretch();
    details->addLayout(fields, 1);
    body->addLayout(details);

    body->addWidget(NoteStrip(tr("Writes into the archive file. Game data stays untouched. "
                                 "Lichii-only artwork lives in Game settings → Artwork."),
                              card));
    AddButtons(tr("Save to file"), QStringLiteral("floppy-disk"),
               FileBadge(nup ? tr("NUP") : tr("NX"), ToQString(archive), card));
    UpdateIconPreview();
    name_edit->setFocus();
}

ArchiveEditDialog::~ArchiveEditDialog() = default;

FileSys::Nup::MetadataEdit ArchiveEditDialog::Edit() const {
    FileSys::Nup::MetadataEdit edit;
    if (remove_icon) {
        edit.icon = std::vector<u8>{};
    } else if (!custom_icon.isEmpty()) {
        edit.icon = std::vector<u8>(custom_icon.begin(), custom_icon.end());
    }
    if (const auto name = name_edit->text().trimmed().toStdString(); name != current.title_name) {
        edit.title_name = name;
    }
    if (const auto publisher = publisher_edit->text().trimmed().toStdString();
        publisher_edit->isModified() && publisher != current.publisher) {
        edit.publisher = publisher;
    }
    if (format == Format::Nx) {
        return edit;
    }
    // Tags the console has no field for stay as stored.
    std::vector<std::pair<std::string, std::string>> localized;
    for (const auto& entry : current.localized_titles) {
        if (!IsEditableLanguage(entry.first)) {
            localized.push_back(entry);
        }
    }
    for (const auto& [tag, field] : localized_edits) {
        if (const auto name = field->text().trimmed().toStdString(); !name.empty()) {
            localized.emplace_back(tag, name);
        }
    }
    auto stored = current.localized_titles;
    auto wanted = localized;
    std::ranges::sort(stored);
    std::ranges::sort(wanted);
    if (stored != wanted) {
        edit.localized_titles = std::move(localized);
    }
    if (const auto version = version_edit->text().trimmed().toStdString();
        version != current.display_version) {
        edit.display_version = version;
    }
    return edit;
}

QString ArchiveEditDialog::DisplayName() const {
    return name_edit->text().trimmed();
}

QPixmap ArchiveEditDialog::CurrentIcon() const {
    return current_icon;
}

void ArchiveEditDialog::ChooseIcon() {
    const auto path = QFileDialog::getOpenFileName(
        this, tr("Choose Icon"), {}, tr("Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"));
    if (!path.isEmpty()) {
        SetIconFromFile(path);
    }
}

void ArchiveEditDialog::UseImage(const QImage& image, bool crop) {
    if (image.isNull()) {
        icon_hint->setText(tr("That image could not be read."));
        return;
    }
    QRect square = CenterSquare(image);
    if (crop) {
        const auto chosen = ChooseIconCrop(image, this);
        if (!chosen) {
            return;
        }
        square = *chosen;
    }
    icon_source = image;
    icon_crop = square;
    remove_icon = false;
    EncodeIcon();
}

void ArchiveEditDialog::SetIconFromFile(const QString& path) {
    UseImage(ReadImage(path), true);
}

void ArchiveEditDialog::FindIcon() {
    icon_search->Start(name_edit->text());
}

void ArchiveEditDialog::OnIconFound(const QByteArray& png, const QString& resolved_title) {
    QImage image;
    image.loadFromData(png);
    UseImage(image, false);
    if (!resolved_title.isEmpty() && !image.isNull()) {
        name_edit->setText(resolved_title);
    }
}

void ArchiveEditDialog::ClearIcon() {
    icon_source = {};
    custom_icon.clear();
    remove_icon = true;
    UpdateIconPreview();
}

void ArchiveEditDialog::RestoreIcon() {
    const bool pending = remove_icon || !custom_icon.isEmpty();
    icon_source = {};
    remove_icon = false;
    custom_icon.clear();
    if (!pending && !original_icon.empty()) {
        custom_icon = QByteArray(reinterpret_cast<const char*>(original_icon.data()),
                                 static_cast<qsizetype>(original_icon.size()));
    }
    UpdateIconPreview();
}

void ArchiveEditDialog::Recrop() {
    if (const auto crop = ChooseIconCrop(icon_source, this)) {
        icon_crop = *crop;
        EncodeIcon();
    }
}

void ArchiveEditDialog::EncodeIcon() {
    if (icon_source.isNull()) {
        return;
    }
    const int size = format == Format::Nx ? 256 << icon_size_index : IconSize;
    custom_icon = EncodeArchiveIcon(icon_source, icon_crop, size);
    if (custom_icon.isEmpty()) {
        icon_hint->setText(tr("That image could not be converted."));
        return;
    }
    UpdateIconPreview();
}

void ArchiveEditDialog::UpdateIconPreview() {
    QPixmap pixmap;
    const QString kind = format == Format::Nup ? tr("NUP") : tr("NX archive");
    const bool pending = remove_icon || !custom_icon.isEmpty();
    if (remove_icon) {
        icon_hint->setText(tr("Removed from the %1 when you save").arg(kind));
    } else if (!custom_icon.isEmpty()) {
        pixmap.loadFromData(custom_icon);
        icon_hint->setText(tr("Custom image · %1 × %1").arg(pixmap.width()));
    } else if (!current.icon.empty()) {
        pixmap = PixmapFromBytes(current.icon);
        icon_hint->setText(tr("Stored in the %1 · %2 × %2").arg(kind).arg(pixmap.width()));
    } else {
        icon_hint->setText(tr("No icon. Drop an image on the square."));
    }
    current_icon = pixmap;
    const bool original = !original_icon.empty() && original_icon != current.icon;
    restore_icon->setVisible(pending || original);
    restore_icon->setToolTip(pending ? tr("Undo the icon change")
                                     : tr("Use the icon the archive was made with"));
    clear_icon->setVisible(format == Format::Nup && !remove_icon &&
                           (!current.icon.empty() || !custom_icon.isEmpty()));
    recrop->setVisible(!icon_source.isNull() && icon_source.width() != icon_source.height());
    icon_preview->SetIcon(pixmap);
}

NupProgressDialog::NupProgressDialog(const QString& operation, const QString& title,
                                     const QPixmap& icon, QWidget* parent)
    : ArchiveCardDialog(operation, 520, parent) {
    setModal(true);
    SetHeader(QStringLiteral("archive"), {});
    step = operation;

    auto* game = new QHBoxLayout;
    game->setSpacing(14);
    if (!icon.isNull()) {
        auto* icon_label = new QLabel(card);
        icon_label->setFixedSize(52, 52);
        icon_label->setPixmap(RoundedPreview(icon, 52, Spotlight::Radius::Row));
        game->addWidget(icon_label);
    }
    auto* name = TextLabel(title.isEmpty() ? operation : title, 16, 600, Spotlight::Color::Text());
    name->setWordWrap(true);
    game->addWidget(name, 1);
    body->addLayout(game);

    auto* progress = new QVBoxLayout;
    progress->setSpacing(10);
    auto* top = new QHBoxLayout;
    top->setSpacing(8);
    status = TextLabel(step, 14, 600, Spotlight::Color::Text());
    percent = TextLabel({}, 14, 600, Spotlight::Color::Text());
    top->addWidget(status, 1);
    top->addWidget(percent);
    progress->addLayout(top);
    bar = new QProgressBar(card);
    bar->setRange(0, 1000);
    bar->setTextVisible(false);
    progress->addWidget(bar);
    auto* bottom = new QHBoxLayout;
    bottom->setSpacing(6);
    details = TextLabel(tr("Preparing…"), 12, 400, Spotlight::Color::TextSecondary());
    remaining = TextLabel({}, 12, 600, QColor(0xff, 0xff, 0xff, 0xb8));
    bottom->addWidget(details, 1);
    bottom->addWidget(remaining);
    progress->addLayout(bottom);
    body->addLayout(progress);

    auto* row = new QHBoxLayout;
    row->addStretch();
    cancel = new Spotlight::PillButton(tr("Cancel"), {}, Spotlight::PillButton::Kind::Control, 40,
                                       card);
    cancel->SetForegroundColor(Spotlight::Color::AccentLight());
    connect(cancel, &QAbstractButton::clicked, this, &NupProgressDialog::CancelRequested);
    row->addWidget(cancel);
    body->addSpacing(6);
    body->addLayout(row);
    timer.start();
}

void NupProgressDialog::SetStep(const QString& new_step) {
    step = new_step;
    status->setText(step);
    percent->clear();
    remaining->clear();
    bar->setValue(0);
    timer.restart();
}

void NupProgressDialog::SetProgress(const FileSys::Nup::PackProgress& progress) {
    if (cancelling) {
        return;
    }
    const double fraction =
        progress.bytes_total == 0 ? 0.0 : double(progress.bytes_done) / progress.bytes_total;
    bar->setValue(static_cast<int>(fraction * 1000));
    percent->setText(QStringLiteral("%1%").arg(fraction * 100, 0, 'f', 0));
    QString text = tr("%1 of %2").arg(HumanSize(progress.bytes_done),
                                      HumanSize(progress.bytes_total));
    if (progress.files_total != 0) {
        text += tr(" · %1 of %2 files").arg(progress.files_done).arg(progress.files_total);
    }
    const double seconds = timer.elapsed() / 1000.0;
    if (seconds >= 3.0 && progress.bytes_done > 0 && fraction < 1.0) {
        const double rate = progress.bytes_done / seconds;
        const int left = static_cast<int>((progress.bytes_total - progress.bytes_done) / rate);
        text += tr(" · %1 MB/s").arg(rate / (1024 * 1024), 0, 'f', 0);
        remaining->setText(left < 60 ? tr("About %1 s left").arg(std::max(left, 1))
                                     : tr("About %1 min left").arg((left + 59) / 60));
    } else {
        remaining->clear();
    }
    details->setText(text);
}

void NupProgressDialog::SetCancelling() {
    cancelling = true;
    cancel->setEnabled(false);
    status->setText(tr("Cancelling…"));
    percent->clear();
    remaining->clear();
    bar->setRange(0, 0);
}

void NupProgressDialog::Finish() {
    finished = true;
    close();
}

void NupProgressDialog::closeEvent(QCloseEvent* event) {
    // Only the window code closes this dialog; a close request cancels the job.
    if (!finished) {
        emit CancelRequested();
        event->ignore();
        return;
    }
    QDialog::closeEvent(event);
}

void NupProgressDialog::reject() {
    if (!finished) {
        emit CancelRequested();
        return;
    }
    QDialog::reject();
}

ArchiveContentsDialog::ArchiveContentsDialog(const QString& archive_name, u64 archive_bytes,
                                             const QPixmap& icon, const std::vector<Entry>& entries,
                                             bool offer_files, QWidget* parent)
    : ArchiveCardDialog(tr("Updates and DLC in this archive"), 520, parent) {
    SetHeader(QStringLiteral("stack"),
              tr("%1 · %2").arg(archive_name, HumanSize(archive_bytes)));
    auto* list = new Box(Spotlight::Color::FillFaint(), Spotlight::Color::StrokeSubtle(),
                         Spotlight::Radius::Panel, card);
    auto* rows = new QVBoxLayout(list);
    rows->setContentsMargins(0, 0, 0, 0);
    rows->setSpacing(0);
    const qreal ratio = devicePixelRatioF();
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (i > 0) {
            auto* line = new Box(Spotlight::Color::StrokeSubtle(), Qt::transparent, 0, list);
            line->setFixedHeight(1);
            rows->addWidget(line);
        }
        auto* row = new QHBoxLayout;
        row->setContentsMargins(12, 10, 12, 10);
        row->setSpacing(12);
        auto* tile = new QLabel(list);
        tile->setFixedSize(36, 36);
        if (entry.kind == Kind::Base && !icon.isNull()) {
            tile->setPixmap(RoundedPreview(icon, 36, Spotlight::Radius::Small));
        } else {
            QPixmap glyph(QSize(36, 36) * ratio);
            glyph.setDevicePixelRatio(ratio);
            glyph.fill(Qt::transparent);
            QPainter painter(&glyph);
            painter.setRenderHint(QPainter::Antialiasing);
            painter.setPen(Qt::NoPen);
            painter.setBrush(Spotlight::Color::Fill());
            painter.drawRoundedRect(QRectF(0, 0, 36, 36), Spotlight::Radius::Small,
                                    Spotlight::Radius::Small);
            painter.drawPixmap(9, 9,
                               Spotlight::IconPixmap(entry.kind == Kind::Dlc ? u"puzzle-piece"
                                                                             : u"arrow-circle-up",
                                                     18, Spotlight::Color::TextSecondary(), ratio));
            painter.end();
            tile->setPixmap(glyph);
        }
        row->addWidget(tile);
        auto* texts = new QVBoxLayout;
        texts->setSpacing(2);
        texts->addWidget(TextLabel(entry.title, 14, 600, Spotlight::Color::Text()));
        texts->addWidget(TextLabel(entry.detail, 12, 400, Spotlight::Color::TextSecondary()));
        row->addLayout(texts, 1);
        if (entry.kind == Kind::Base) {
            auto* badge = new Box(Spotlight::Color::Fill(), Qt::transparent, 11, list);
            auto* badge_layout = new QHBoxLayout(badge);
            badge_layout->setContentsMargins(9, 0, 9, 0);
            badge_layout->addWidget(TextLabel(tr("Base"), 11, 600, Spotlight::Color::Text()));
            badge->setFixedHeight(22);
            row->addWidget(badge);
        } else {
            auto* remove = new Spotlight::CircleButton(QStringLiteral("trash"), 30, list);
            remove->setToolTip(tr("Remove from the archive"));
            remove->setAccessibleName(tr("Remove %1").arg(entry.title));
            connect(remove, &QAbstractButton::clicked, this, [this, i] {
                removed = i;
                choice = Choice::Remove;
                accept();
            });
            row->addWidget(remove);
        }
        rows->addLayout(row);
    }
    body->addWidget(list);

    auto* adds = new QHBoxLayout;
    adds->setSpacing(10);
    auto* add_folder =
        new FlatButton(tr("Add update folder…"), QStringLiteral("folder-open"), true, 34, card);
    connect(add_folder, &QAbstractButton::clicked, this, [this] {
        choice = Choice::AddFolder;
        accept();
    });
    adds->addWidget(add_folder);
    if (offer_files) {
        auto* add_files =
            new FlatButton(tr("Add NSP or XCI…"), QStringLiteral("file"), true, 34, card);
        connect(add_files, &QAbstractButton::clicked, this, [this] {
            choice = Choice::AddFiles;
            accept();
        });
        adds->addWidget(add_files);
    }
    adds->addStretch();
    body->addLayout(adds);

    auto* note = new Box(QColor(0xff, 0xff, 0xff, 0x08), Spotlight::Color::StrokeSubtle(),
                         Spotlight::Radius::Tile, card);
    auto* note_layout = new QHBoxLayout(note);
    note_layout->setContentsMargins(14, 12, 14, 12);
    note_layout->setSpacing(10);
    auto* info = new QLabel(note);
    info->setPixmap(Spotlight::IconPixmap(u"info", 16, Spotlight::Color::TextSecondary(), ratio));
    note_layout->addWidget(info, 0, Qt::AlignTop);
    auto* note_text = TextLabel(
        tr("Changes rewrite the archive in place. Close the game first if it is running."), 13,
        400, QColor(0xff, 0xff, 0xff, 0xb8));
    note_text->setWordWrap(true);
    note_layout->addWidget(note_text, 1);
    body->addWidget(note);

    auto* footer = new QHBoxLayout;
    footer->addStretch();
    action_button = new Spotlight::PillButton(tr("Done"), {}, Spotlight::PillButton::Kind::Accent,
                                              40, card);
    connect(action_button, &QAbstractButton::clicked, this, &QDialog::accept);
    footer->addWidget(action_button);
    body->addSpacing(6);
    body->addLayout(footer);
}

NupExtractDialog::NupExtractDialog(const std::filesystem::path& archive_,
                                   const FileSys::Nup::Metadata& probed, QWidget* parent)
    : ArchiveCardDialog(tr("Extract to a folder"), 520, parent), archive{archive_},
      target{FileSys::Nup::DefaultExtractPath(archive_)} {
    SetHeader(QStringLiteral("folder-open"),
              tr("Unpacks the archive into loose files. The .nup stays as it is."));
    std::error_code ec;
    needed = std::filesystem::file_size(archive, ec);
    if (ec) {
        needed = 0;
    }
    auto* game = new QHBoxLayout;
    game->setSpacing(14);
    const QPixmap icon = PixmapFromBytes(probed.icon);
    if (!icon.isNull()) {
        auto* icon_label = new QLabel(card);
        icon_label->setFixedSize(46, 46);
        icon_label->setPixmap(RoundedPreview(icon, 46, Spotlight::Radius::Field));
        game->addWidget(icon_label);
    }
    auto* texts = new QVBoxLayout;
    texts->setSpacing(2);
    const QString name = QString::fromStdString(probed.title_name).trimmed();
    texts->addWidget(TextLabel(name.isEmpty() ? ToQString(archive.stem()) : name, 15, 600,
                               Spotlight::Color::Text()));
    texts->addWidget(TextLabel(tr("%1 · %2").arg(ToQString(archive.filename()), HumanSize(needed)),
                               13, 400, Spotlight::Color::TextSecondary()));
    game->addLayout(texts, 1);
    body->addLayout(game);

    auto* destination = new QVBoxLayout;
    destination->setSpacing(10);
    destination->addWidget(
        TextLabel(tr("Destination"), 13, 600, Spotlight::Color::TextSecondary()));
    auto* row = new QHBoxLayout;
    row->setSpacing(10);
    auto* path = new PathRow(u"folder", card);
    target_row = path;
    auto* browse = new FlatButton(tr("Browse…"), {}, true, 40, card);
    connect(browse, &QAbstractButton::clicked, this, &NupExtractDialog::ChooseTarget);
    row->addWidget(path, 1);
    row->addWidget(browse);
    destination->addLayout(row);
    space_label = Hint({});
    destination->addWidget(space_label);
    body->addLayout(destination);
    AddButtons(tr("Extract"), QStringLiteral("download-simple"));
    UpdateSpace();
}

std::filesystem::path NupExtractDialog::Target() const {
    return target;
}

void NupExtractDialog::ChooseTarget() {
    const QString chosen = QFileDialog::getSaveFileName(this, tr("Extract to"), ToQString(target));
    if (chosen.isEmpty()) {
        return;
    }
    target = FromQString(chosen);
    UpdateSpace();
}

void NupExtractDialog::UpdateSpace() {
    static_cast<PathRow*>(target_row)->SetPath(ToQString(target));
    const u64 available = Common::FS::GetFreeSpaceSize(target.parent_path());
    std::error_code ec;
    QString problem;
    if (std::filesystem::exists(target, ec)) {
        problem = tr("This folder already exists.");
    } else if (available < needed) {
        problem = tr("Not enough free space on this drive.");
    }
    QString text =
        tr("Needs about %1 · %2 free").arg(HumanSize(needed), HumanSize(available));
    if (!problem.isEmpty()) {
        text = problem + QLatin1Char(' ') + text;
    }
    space_label->setText(text);
    Spotlight::SetTextColor(space_label, problem.isEmpty() ? Spotlight::Color::TextSecondary()
                                                           : Spotlight::Color::Warning());
    action_button->setEnabled(problem.isEmpty());
}

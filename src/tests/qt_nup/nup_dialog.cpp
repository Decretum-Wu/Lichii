// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QFile>
#include <QLineEdit>
#include <QTemporaryDir>
#include <QTest>
#include <QtEndian>

#include "lichii/nup_dialogs.h"

namespace {

QByteArray ReadRecord(const QString& path, int package, quint16 type) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly) || !file.seek(0x80 + 0x30 + package * 0x40 + 0x20)) {
        return {};
    }
    const auto extent = file.read(12);
    if (extent.size() != 12 || !file.seek(qFromLittleEndian<quint64>(extent.constData()))) {
        return {};
    }
    const auto block = file.read(qFromLittleEndian<quint32>(extent.constData() + 8));
    for (qsizetype pos = 0; pos + 8 <= block.size();) {
        const auto* header = block.constData() + pos;
        const qsizetype size = qFromLittleEndian<quint32>(header + 4);
        if (size > block.size() - pos - 8) {
            return {};
        }
        if (qFromLittleEndian<quint16>(header) == type) {
            return block.mid(pos, 8 + size);
        }
        pos += 8 + size;
    }
    return {};
}

class NupDialogTest : public QObject {
    Q_OBJECT

private slots:
    void renamePreservesBuildIds();
};

void NupDialogTest::renamePreservesBuildIds() {
    const auto fixture = QFINDTESTDATA("shifted_base_update.nup");
    QVERIFY(!fixture.isEmpty());
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const auto path = temp.filePath(QStringLiteral("edited.nup"));
    QVERIFY(QFile::copy(fixture, path));
    const auto archive = std::filesystem::path(path.toStdU16String());
    const auto base_id = ReadRecord(fixture, 0, 0x0006);
    const auto update_id = ReadRecord(fixture, 1, 0x0006);
    QCOMPARE(base_id.size(), 40);
    QCOMPARE(update_id.size(), 40);
    QVERIFY(base_id != update_id);

    NupEditDialog dialog(archive);
    auto* name = dialog.findChild<QLineEdit*>(QStringLiteral("nupEditName"));
    QVERIFY(name);
    QTest::keyClicks(name, QStringLiteral("Edited title"));
    const auto edit = dialog.Edit();
    QVERIFY(edit.title_name == std::string("Edited title"));
    std::string error;
    QVERIFY2(FileSys::Nup::EditNupMetadata(archive, edit, &error) ==
                 FileSys::Nup::EditResult::Success,
             error.c_str());
    QCOMPARE(ReadRecord(path, 0, 0x0006), base_id);
    QCOMPARE(ReadRecord(path, 1, 0x0006), update_id);

    auto* publisher = dialog.findChild<QLineEdit*>(QStringLiteral("nupEditPublisher"));
    QVERIFY(publisher);
    publisher->selectAll();
    QTest::keyClicks(publisher, QStringLiteral("Edited publisher"));
    QVERIFY(dialog.Edit().publisher == std::string("Edited publisher"));
}

} // namespace

QTEST_MAIN(NupDialogTest)
#include "nup_dialog.moc"

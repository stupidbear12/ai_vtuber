import "./globals.css";

export const metadata = {
  title: "시온(sion) - AI DJ VTuber",
  description: "시온과 대화해보세요! 밝고 친근한 AI DJ VTuber 시온입니다.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body className="bg-sion-bg text-sion-text antialiased">
        {children}
      </body>
    </html>
  );
}

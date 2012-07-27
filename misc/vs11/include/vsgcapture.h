#pragma once
#pragma comment(lib,"VsGraphicsHelper.lib")

#include <Windows.h>
#include <stdio.h>
#include <sal.h>

#ifndef VSG_DEFAULT_RUN_FILENAME
#define VSG_DEFAULT_RUN_FILENAME L"default.vsglog"
#endif

extern "C" void __stdcall VsgDbgInit(wchar_t const * szVSGLog);
extern "C" void __stdcall VsgDbgUnInit();
extern "C" void __stdcall VsgDbgToggleHUD();
extern "C" void __stdcall VsgDbgCaptureCurrentFrame();

class VsgDbg
{
public:
	VsgDbg(bool bDefaultInit)
	{
		if(bDefaultInit)
		{
#ifndef DONT_SAVE_VSGLOG_TO_TEMP	

#if WINAPI_FAMILY == 2
			Init(Platform::String::Concat(Platform::String::Concat(Windows::Storage::ApplicationData::Current->TemporaryFolder->Path, L"\\"), VSG_DEFAULT_RUN_FILENAME)->Data());
#else
			wchar_t tempDir[MAX_PATH];
			wchar_t filePath[MAX_PATH];

			if(GetTempPath(MAX_PATH, tempDir) == 0)
			{
				return;
			}

			swprintf_s(filePath, MAX_PATH, L"%s%s", tempDir, VSG_DEFAULT_RUN_FILENAME);
			Init(filePath);
#endif

#else
			Init(VSG_DEFAULT_RUN_FILENAME);
#endif
		}
	}

	~VsgDbg()
	{
		UnInit();
	}

	void Init(_In_z_ wchar_t const * szVSGLog)
	{
		VsgDbgInit(szVSGLog);
	}

	void UnInit()
	{
		VsgDbgUnInit();
	}

	void ToggleHUD()
	{
		VsgDbgToggleHUD();
	}

	void CaptureCurrentFrame ()
	{
		VsgDbgCaptureCurrentFrame ();
	}
};

#ifndef VSG_NODEFAULT_INSTANCE
	_declspec(selectany) VsgDbg *g_pVsgDbg;
	
	inline void  UnInitVsPix()
	{
		if(g_pVsgDbg != NULL)
		{
			delete g_pVsgDbg;
		}
	}

	inline void  InitVsPix()
	{
		g_pVsgDbg = new VsgDbg(true); atexit(&UnInitVsPix); 
	}


	#pragma section(".CRT$XCT",long,read)
	__declspec(allocate(".CRT$XCT"))  _declspec(dllexport) void (*pInitFunc)() = InitVsPix;
		
#endif